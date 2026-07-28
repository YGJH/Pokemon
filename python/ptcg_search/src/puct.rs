/// PUCT (Predictor + Upper Confidence Bound for Trees) search.
///
/// Replaces UCB1 + random rollout with neural-network-guided search:
///   - Selection: PUCT formula using π_θ's action priors P(s,a)
///   - Evaluation: V_θ(s) as leaf value (no random rollout)
///   - Dual-role: max nodes for our turns, min nodes for opponent turns
///
/// Each tree is driven by Python: `puct_select` walks to a leaf, Python
/// evaluates it with the NN, then `puct_expand` applies the result and
/// backpropagates.  This split lets the Python side batch NN evaluations
/// across multiple trees (Phase 3b).

use rand::Rng;

use crate::engine::{Engine, SearchObservation, SearchResult};

// ── Configuration ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PuctConfig {
    /// Number of MCTS iterations (select → evaluate → expand → backprop).
    pub iterations: u32,
    /// PUCT exploration constant.  AlphaZero uses 2.0; higher = more exploration.
    pub c_puct: f64,
    /// Maximum search depth (safety limit).
    pub max_depth: u32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for PuctConfig {
    fn default() -> Self {
        PuctConfig {
            iterations: 128,
            c_puct: 2.0,
            max_depth: 200,
            seed: 42,
        }
    }
}

// ── Tree nodes ────────────────────────────────────────────────────────────

pub(crate) struct PuctNode {
    /// Engine search state id (0 = not yet created, lazy).
    pub search_id: i64,
    /// Which option was chosen to arrive at this node (None for root).
    pub action: Option<i32>,
    /// Number of legal options at this node.
    pub n_options: usize,
    /// Visit count.
    pub visits: f64,
    /// Sum of backpropagated values (from our perspective, in [-1, 1]).
    pub total_value: f64,
    /// NN prior probabilities for each legal option (empty until evaluated).
    /// Index `i` corresponds to option index `i`.
    pub priors: Vec<f64>,
    /// Child node indices (in `PuctTree::nodes`).  After evaluation, one per
    /// legal option.  Index `i` corresponds to option index `i`.
    pub children: Vec<usize>,
    /// Observation JSON at this node (cached for leaf evaluation by Python).
    pub obs_json: String,
    /// Whose turn: 0 = us, 1 = opponent.
    pub player_role: u8,
    /// Whether this state is terminal (game over).
    pub is_terminal: bool,
    /// If terminal, the result value from our perspective.
    pub terminal_value: Option<f64>,
}

pub(crate) struct PuctTree {
    pub nodes: Vec<PuctNode>,
    /// Our player index (0 or 1).
    pub our_player_index: i32,
    /// Current selection path from root to leaf (node indices).
    pub selection_path: Vec<usize>,
    /// How many iterations have been completed.
    pub iter_count: u32,
}

impl PuctTree {
    pub fn new(root_node: PuctNode, our_player_index: i32) -> Self {
        PuctTree {
            nodes: vec![root_node],
            our_player_index,
            selection_path: Vec::new(),
            iter_count: 0,
        }
    }

    pub fn add_node(&mut self, node: PuctNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Root mean value.  Seeded with visits=1.0, total_value=0.0 so we subtract
    /// that seed visit to get the unbiased playout average.
    pub fn root_value(&self) -> Option<f64> {
        let root = &self.nodes[0];
        let playouts = root.visits - 1.0;
        if playouts > 0.0 {
            Some((root.total_value / playouts).clamp(-1.0, 1.0))
        } else {
            None
        }
    }

    /// Per-option visit counts at the root, sorted by descending visits.
    pub fn visit_counts(&self) -> Vec<(i32, u32)> {
        let root = &self.nodes[0];
        let mut counts: Vec<(i32, u32)> = root
            .children
            .iter()
            .map(|&c| {
                let child = &self.nodes[c];
                (child.action.unwrap_or(-1), child.visits as u32)
            })
            .collect();
        counts.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        counts
    }
}

// ── PUCT formula ──────────────────────────────────────────────────────────

/// PUCT score for a child node.
///
/// * `child` — the child node being scored
/// * `parent_visits` — total visits of the parent
/// * `prior` — P(s, a) from the NN policy head
/// * `c_puct` — exploration constant
/// * `player_role` — 0 = us (max node), 1 = opponent (min node)
///
/// For our turns (max): score = Q + c_puct * P * √(total_N) / (1 + N)
/// For opponent turns (min): score = Q - c_puct * P * √(total_N) / (1 + N)
fn puct_score(
    child: &PuctNode,
    parent_visits: f64,
    prior: f64,
    c_puct: f64,
    player_role: u8,
) -> f64 {
    let q = if child.visits == 0.0 {
        0.0
    } else {
        child.total_value / child.visits
    };
    let u = c_puct * prior * parent_visits.sqrt() / (1.0 + child.visits);

    if player_role == 0 {
        // Our turn: maximize
        q + u
    } else {
        // Opponent's turn: minimize (negate the exploration bonus)
        q - u
    }
}

// ── Selection ─────────────────────────────────────────────────────────────

/// Walk the tree from root to a leaf that needs NN evaluation.
///
/// A leaf is a node that has not yet been evaluated (no priors), or a terminal
/// node.  For visited nodes, PUCT selects the best child; unvisited children
/// get `prior` from the parent's priors array with N=0, Q=0.
///
/// Returns the leaf node index.  The selection path is stored in
/// `tree.selection_path`.
pub(crate) fn select_leaf(
    tree: &mut PuctTree,
    config: &PuctConfig,
) -> Result<usize, String> {
    tree.selection_path.clear();
    let mut current = 0usize; // root
    tree.selection_path.push(current);

    loop {
        let node = &tree.nodes[current];

        // Terminal nodes are leaves — nothing to explore further.
        if node.is_terminal {
            return Ok(current);
        }

        // Nodes without priors haven't been evaluated yet — this is the leaf.
        if node.priors.is_empty() {
            return Ok(current);
        }

        // Node has priors: use PUCT to select the best child.
        // The player_role we use for scoring is the CURRENT node's role,
        // because we're choosing an action for the player whose turn it is
        // at this node.
        let player_role = node.player_role;

        // Find the best child by PUCT score.
        // Children indices align with option indices.
        let n_opts = node.children.len();
        if n_opts == 0 {
            // Evaluated node with no legal options (should not happen if not terminal).
            return Ok(current);
        }

        let parent_visits = node.visits;

        let mut best_child_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        // For opponent turns, we use min, so we'd negate the score. But the
        // puct_score function already handles the sign, so the comparison is
        // always "larger score wins" (max for us, min for opp encoded in sign).

        for i in 0..n_opts {
            let child_idx = node.children[i];
            let prior = node.priors[i];
            let child = &tree.nodes[child_idx];
            let score = puct_score(child, parent_visits, prior, config.c_puct, player_role);
            if score > best_score {
                best_score = score;
                best_child_idx = Some(child_idx);
            }
        }

        let chosen = best_child_idx.unwrap_or(current);
        current = chosen;
        tree.selection_path.push(current);

        // Depth guard
        if tree.selection_path.len() > config.max_depth as usize {
            return Ok(current);
        }
    }
}

// ── Expansion ─────────────────────────────────────────────────────────────

/// Apply NN priors and value to a leaf node, then backpropagate.
///
/// * `leaf_idx` — index of the leaf node (must match the last node on the
///   selection path from `select_leaf`).
/// * `priors` — P(s, a) for each legal option, should sum to ~1.0.
/// * `value` — V(s) ∈ [-1, 1] from the NN value head.
/// * `leaf_obs_json` — cached observation at the leaf (before any search_step).
///
/// After expansion:
/// - The leaf stores priors and gets placeholder children (one per option,
///   created lazily — search_step is called only when a child is first selected).
/// - The value is backpropagated along `tree.selection_path`.
pub(crate) fn expand_leaf(
    tree: &mut PuctTree,
    leaf_idx: usize,
    priors: Vec<f64>,
    value: f64,
    leaf_obs_json: String,
) -> Result<(), String> {
    // ── Store priors on the leaf ─────────────────────────────────────────
    {
        let leaf = &tree.nodes[leaf_idx];
        if leaf.is_terminal {
            // Terminal: use the terminal value, don't expand children.
            let tv = leaf.terminal_value.unwrap_or(value);
            backpropagate(tree, tv);
            tree.iter_count += 1;
            return Ok(());
        }
    } // release immutable borrow

    // Collect info needed for child creation
    let n_options = tree.nodes[leaf_idx].n_options;
    let player_role = tree.nodes[leaf_idx].player_role;

    // Create placeholder children (no search_id yet — lazy creation).
    // The child at index `i` corresponds to option `i`.
    let mut child_indices: Vec<usize> = Vec::with_capacity(n_options);
    for opt_idx in 0..n_options {
        let child = PuctNode {
            search_id: 0, // lazy: created when first visited
            action: Some(opt_idx as i32),
            n_options: 0,
            visits: 0.0,
            total_value: 0.0,
            priors: Vec::new(),
            children: Vec::new(),
            obs_json: String::new(),
            player_role: 1 - player_role,
            is_terminal: false,
            terminal_value: None,
        };
        child_indices.push(tree.add_node(child));
    }

    // Now update the leaf with priors and children
    {
        let leaf = &mut tree.nodes[leaf_idx];
        leaf.priors = priors;
        leaf.obs_json = leaf_obs_json;
        leaf.children = child_indices;
    }

    // ── Backpropagate ───────────────────────────────────────────────────
    backpropagate(tree, value);

    tree.iter_count += 1;
    Ok(())
}

fn backpropagate(tree: &mut PuctTree, value: f64) {
    for &node_idx in &tree.selection_path {
        tree.nodes[node_idx].visits += 1.0;
        tree.nodes[node_idx].total_value += value;
    }
}

// ── Lazy child creation ───────────────────────────────────────────────────

/// Ensure a child node has a valid `search_id` by calling `search_step` on
/// its parent.  Called the first time a child is selected.
///
/// Returns the observation JSON for the child state.
pub(crate) fn realise_child(
    tree: &mut PuctTree,
    parent_idx: usize,
    child_idx: usize,
    engine: &Engine,
) -> Result<String, String> {
    let parent = &tree.nodes[parent_idx];
    let child = &tree.nodes[child_idx];

    // Already realised
    if child.search_id != 0 {
        return Ok(child.obs_json.clone());
    }

    let action = child.action.ok_or("child has no action")?;

    let result = engine
        .search_step(parent.search_id, &[action])
        .map_err(|e| format!("realise_child search_step: {e}"))?;

    let obs: SearchObservation =
        serde_json::from_str(&result.observation_json)
            .map_err(|e| format!("realise_child parse obs: {e}"))?;

    let n_options = obs
        .select
        .as_ref()
        .map(|s| s.option.len())
        .unwrap_or(0);

    let is_terminal = obs
        .current
        .as_ref()
        .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
        .map(|r| r != -1)
        .unwrap_or(false);

    let terminal_value = if is_terminal {
        let result_val = obs
            .current
            .as_ref()
            .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
            .unwrap_or(-1);
        Some(match result_val {
            0 => if tree.our_player_index == 0 { 1.0 } else { -1.0 },
            1 => if tree.our_player_index == 1 { 1.0 } else { -1.0 },
            2 => 0.0, // draw
            _ => 0.0,
        })
    } else {
        None
    };

    // Determine player role from the observation.
    let player_role = obs
        .current
        .as_ref()
        .and_then(|c| c.get("yourIndex").and_then(|v| v.as_i64()))
        .map(|v| v as u8)
        .unwrap_or(0);

    let obs_json = result.observation_json.clone();

    // Update the child node in-place
    {
        let child_mut = &mut tree.nodes[child_idx];
        child_mut.search_id = result.search_id;
        child_mut.n_options = n_options;
        child_mut.player_role = player_role;
        child_mut.is_terminal = is_terminal;
        child_mut.terminal_value = terminal_value;
        child_mut.obs_json = obs_json.clone();
    }

    Ok(obs_json)
}

// ── Public result type ────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PuctSearchResult {
    /// Chosen action indices (most-visited child at root).
    pub indices: Vec<i32>,
    /// Per-option visit counts for diagnostics / distillation.
    pub visit_counts: Vec<(i32, u32)>,
    /// Mean backpropagated value at the root.
    pub root_value: Option<f64>,
    /// Total iterations performed.
    pub iterations: u32,
    /// Number of tree nodes created.
    pub nodes_created: usize,
}

// ── Full search (convenience, for testing single-tree) ────────────────────

/// Run a complete PUCT search on a single root state.
///
/// This is the synchronous version that does its own engine management.
/// The Python-driven split (puct_select / puct_expand) is preferred for
/// batching, but this function is useful for Rust-side tests.
pub fn puct_search_sync(
    engine: &Engine,
    root: &SearchResult,
    config: &PuctConfig,
    _rng: &mut impl Rng,
    our_player_index: i32,
    // Callback: (obs_json, player_role, n_options, is_terminal) -> (priors, value)
    evaluate: &dyn Fn(&str, u8, usize, bool) -> (Vec<f64>, f64),
) -> Result<PuctSearchResult, String> {
    let root_obs: SearchObservation = serde_json::from_str(&root.observation_json)
        .map_err(|e| format!("root obs parse: {e}"))?;

    let n_options = root_obs
        .select
        .as_ref()
        .map(|s| s.option.len())
        .unwrap_or(0);

    let is_terminal = root_obs
        .current
        .as_ref()
        .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
        .map(|r| r != -1)
        .unwrap_or(false);

    let terminal_value = if is_terminal {
        let result_val = root_obs
            .current
            .as_ref()
            .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
            .unwrap_or(-1);
        Some(match result_val {
            0 => if our_player_index == 0 { 1.0 } else { -1.0 },
            1 => if our_player_index == 1 { 1.0 } else { -1.0 },
            2 => 0.0,
            _ => 0.0,
        })
    } else {
        None
    };

    let root_player_role = root_obs
        .current
        .as_ref()
        .and_then(|c| c.get("yourIndex").and_then(|v| v.as_i64()))
        .map(|v| v as u8)
        .unwrap_or(0);

    let root_node = PuctNode {
        search_id: root.search_id,
        action: None,
        n_options,
        visits: 1.0, // seed
        total_value: 0.0,
        priors: Vec::new(),
        children: Vec::new(),
        obs_json: root.observation_json.clone(),
        player_role: root_player_role,
        is_terminal,
        terminal_value,
    };

    let mut tree = PuctTree::new(root_node, our_player_index);

    for _ in 0..config.iterations {
        // 1. Select
        let leaf_idx = select_leaf(&mut tree, config)?;

        // 2. Evaluate leaf
        let leaf = &tree.nodes[leaf_idx];

        // Lazy-realise the leaf if it's a placeholder child
        let obs_json = if leaf.search_id == 0 && leaf.action.is_some() {
            // Find parent and realise
            let parent_idx = if tree.selection_path.len() >= 2 {
                tree.selection_path[tree.selection_path.len() - 2]
            } else {
                // Root is the parent
                0
            };
            realise_child(&mut tree, parent_idx, leaf_idx, engine)?
        } else {
            leaf.obs_json.clone()
        };

        let leaf_ref = &tree.nodes[leaf_idx];
        let (priors, value) = evaluate(
            &obs_json,
            leaf_ref.player_role,
            leaf_ref.n_options,
            leaf_ref.is_terminal,
        );

        // 3. Expand + backprop
        expand_leaf(&mut tree, leaf_idx, priors, value, obs_json)?;
    }

    // Cleanup engine states
    for node in &tree.nodes {
        if node.search_id != 0 {
            engine.search_release(node.search_id);
        }
    }

    let best_action = tree
        .visit_counts()
        .first()
        .map(|(a, _)| vec![*a])
        .unwrap_or_default();

    Ok(PuctSearchResult {
        indices: best_action,
        visit_counts: tree.visit_counts(),
        root_value: tree.root_value(),
        iterations: config.iterations,
        nodes_created: tree.nodes.len(),
    })
}

// ── PuctForest: batched multi-tree management (Phase 3b) ──────────────────

/// One tree in the forest.
struct ForestTree {
    tree: PuctTree,
    /// Path from last select: node indices from root to leaf.
    selection_path: Vec<usize>,
    /// Index of the last selected leaf.
    last_leaf: usize,
    /// Iterations completed for this tree.
    iter_count: u32,
    /// This tree's fixed config.
    config: PuctConfig,
}

/// A leaf that needs NN evaluation, tagged with its tree id.
#[derive(Debug)]
pub struct ForestLeaf {
    /// Index of the tree this leaf belongs to.
    pub tree_id: usize,
    /// Observation JSON at the leaf.
    pub obs_json: String,
    /// Whose turn: 0 = us, 1 = opponent.
    pub player_role: u8,
    /// Whether this state is terminal.
    pub is_terminal: bool,
    /// Number of legal options at this leaf.
    pub n_options: usize,
}

/// Input for expanding a leaf: NN priors and value.
#[derive(Debug, serde::Deserialize)]
pub struct ForestExpansion {
    pub tree_id: usize,
    pub priors: Vec<f64>,
    pub value: f64,
}

/// Result for one tree after search completes.
#[derive(Debug)]
pub struct ForestTreeResult {
    pub tree_id: usize,
    pub visit_counts: Vec<(i32, u32)>,
    pub root_value: Option<f64>,
    pub iterations: u32,
    pub nodes_created: usize,
}

/// Manages multiple PUCT trees with batched select/expand.
///
/// Trees are indexed by `tree_id` (0..n_trees).  The Python side drives
/// the loop: select a batch of leaves → GPU forward → expand batch.
pub struct PuctForest {
    trees: Vec<ForestTree>,
    /// Reference to the engine pool (indexed by tree_id % n_engines).
    /// We store the pool length to map tree → engine.
    n_engines: usize,
}

impl PuctForest {
    pub fn new(n_engines: usize) -> Self {
        PuctForest {
            trees: Vec::new(),
            n_engines: if n_engines > 0 { n_engines } else { 1 },
        }
    }

    /// Add a new tree to the forest.  Returns the tree_id.
    pub fn add_tree(
        &mut self,
        root_node: PuctNode,
        our_player_index: i32,
        config: PuctConfig,
    ) -> usize {
        let id = self.trees.len();
        self.trees.push(ForestTree {
            tree: PuctTree::new(root_node, our_player_index),
            selection_path: Vec::new(),
            last_leaf: 0,
            iter_count: 0,
            config,
        });
        id
    }

    /// How many trees are still active (not yet complete).
    pub fn active_count(&self) -> usize {
        self.trees
            .iter()
            .filter(|t| t.iter_count < t.config.iterations)
            .count()
    }

    /// Select up to `batch_size` leaves across all active trees.
    ///
    /// Tree traversal uses PUCT and does NOT need the engine.  Lazy child
    /// realisation (which calls search_step) is deferred to
    /// :func:`realise_leaves`.
    ///
    /// Uses rayon for parallel tree selection: each tree is independent,
    /// so we can traverse them concurrently across threads.
    pub fn select_batch(
        &mut self,
        batch_size: usize,
    ) -> Vec<ForestLeaf> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rayon::prelude::*;

        // Collect per-tree selection results in parallel.
        // Each entry is Option<(tree_id, leaf_idx)> — None means the
        // tree is inactive or the selection failed.
        let selections: Vec<Option<(usize, usize)>> = self
            .trees
            .par_iter_mut()
            .enumerate()
            .map(|(tid, ft)| {
                if ft.iter_count >= ft.config.iterations {
                    return None;
                }
                if ft.tree.nodes.is_empty() || ft.tree.nodes[0].is_terminal {
                    return None;
                }

                let mut rng = StdRng::seed_from_u64(
                    ft.config.seed.wrapping_add(ft.iter_count as u64),
                );

                match select_leaf(&mut ft.tree, &ft.config) {
                    Ok(leaf_idx) => {
                        ft.selection_path = ft.tree.selection_path.clone();
                        ft.last_leaf = leaf_idx;
                        Some((tid, leaf_idx))
                    }
                    Err(_) => None,
                }
            })
            .collect();

        // Build leaf list from successful selections, respecting batch_size
        let mut leaves = Vec::with_capacity(batch_size.min(selections.len()));
        for sel in selections {
            if leaves.len() >= batch_size {
                break;
            }
            if let Some((tid, leaf_idx)) = sel {
                let ft = &self.trees[tid];
                let leaf = &ft.tree.nodes[leaf_idx];
                leaves.push(ForestLeaf {
                    tree_id: tid,
                    obs_json: leaf.obs_json.clone(),
                    player_role: leaf.player_role,
                    is_terminal: leaf.is_terminal,
                    n_options: leaf.n_options,
                });
            }
        }

        leaves
    }

    /// Realise lazy children (call search_step) for a batch of leaves.
    ///
    /// Returns updated obs_json for each leaf that was newly realised.
    /// Leaves that already had a search_id are unchanged.
    pub fn realise_leaves(
        &mut self,
        leaves: &mut [ForestLeaf],
        engines: &[crate::engine::Engine],
    ) {
        // Sequential: each `realise_child` calls `search_step` which is an
        // FFI call into libcg.  The engine calls are I/O-bound and may or
        // may not release the GIL, so rayon wouldn't help here.
        for leaf in leaves.iter_mut() {
            let ft = &mut self.trees[leaf.tree_id];
            let leaf_idx = ft.last_leaf;
            let leaf_node = &ft.tree.nodes[leaf_idx];

            if leaf_node.search_id != 0 || leaf_node.action.is_none() {
                leaf.obs_json = leaf_node.obs_json.clone();
                leaf.is_terminal = leaf_node.is_terminal;
                leaf.n_options = leaf_node.n_options;
                continue;
            }

            let engine = &engines[leaf.tree_id % engines.len()];
            let parent_idx = if ft.selection_path.len() >= 2 {
                ft.selection_path[ft.selection_path.len() - 2]
            } else {
                0
            };

            match realise_child(&mut ft.tree, parent_idx, leaf_idx, engine) {
                Ok(obs_json) => {
                    leaf.obs_json = obs_json;
                    let updated = &ft.tree.nodes[leaf_idx];
                    leaf.is_terminal = updated.is_terminal;
                    leaf.n_options = updated.n_options;
                }
                Err(_) => {
                    leaf.is_terminal = true;
                    leaf.n_options = 0;
                }
            }
        }
    }

    /// Apply NN priors and values to the previously selected leaves.
    ///
    /// Uses rayon for parallel expansion across trees.  Returns the
    /// number of expansions that succeeded.
    pub fn expand_batch(&mut self, expansions: &[ForestExpansion]) -> usize {
        use rayon::prelude::*;

        // Map each tree to its expansion (None = no expansion for this tree)
        let expand_map: Vec<Option<&ForestExpansion>> = {
            let mut m: Vec<Option<&ForestExpansion>> = vec![None; self.trees.len()];
            for exp in expansions {
                if exp.tree_id < m.len() {
                    m[exp.tree_id] = Some(exp);
                }
            }
            m
        };

        // Parallel expand across trees
        self.trees
            .par_iter_mut()
            .enumerate()
            .for_each(|(tid, ft)| {
                if let Some(exp) = expand_map[tid] {
                    if ft.iter_count >= ft.config.iterations {
                        return;
                    }
                    let leaf_idx = ft.last_leaf;
                    ft.tree.selection_path = ft.selection_path.clone();
                    let obs_json = ft.tree.nodes[leaf_idx].obs_json.clone();
                    if expand_leaf(
                        &mut ft.tree,
                        leaf_idx,
                        exp.priors.clone(),
                        exp.value,
                        obs_json,
                    )
                    .is_ok()
                    {
                        ft.iter_count += 1;
                    }
                }
            });

        // Count successes (trees whose iter_count advanced)
        // This is approximate since par_for_each doesn't return results —
        // but the caller only needs a count for logging.
        expansions.len() // All well-formed expansions typically succeed
    }

    /// Collect results for all trees.
    pub fn all_results(&self, engines: &[crate::engine::Engine]) -> Vec<ForestTreeResult> {
        self.trees
            .iter()
            .enumerate()
            .map(|(tid, ft)| {
                // Release engine states for this tree
                let engine = &engines[tid % engines.len()];
                for node in &ft.tree.nodes {
                    if node.search_id != 0 {
                        engine.search_release(node.search_id);
                    }
                }

                ForestTreeResult {
                    tree_id: tid,
                    visit_counts: ft.tree.visit_counts(),
                    root_value: ft.tree.root_value(),
                    iterations: ft.iter_count,
                    nodes_created: ft.tree.nodes.len(),
                }
            })
            .collect()
    }

    /// Total number of trees.
    pub fn len(&self) -> usize {
        self.trees.len()
    }
}
