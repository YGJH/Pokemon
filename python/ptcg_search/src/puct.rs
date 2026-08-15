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

    /// Per-root-child `(option_index, mean_value, visits, player_role)`.
    ///
    /// Diagnostic only.  Visit counts alone cannot say *why* the search
    /// preferred an option: a child can lead on visits because its Q is high
    /// or because its prior is, and those call for opposite fixes.  `Q` is in
    /// the root player's frame (`expand_leaf` orients before backup), so these
    /// are directly comparable across children, and `player_role` exposes
    /// whether the child is a node the opponent moves at — the asymmetry that
    /// makes "end the turn" structurally unlike every other option.
    pub fn child_stats(&self) -> Vec<(i32, f64, u32, u8)> {
        let root = &self.nodes[0];
        root.children
            .iter()
            .map(|&c| {
                let child = &self.nodes[c];
                let q = if child.visits > 0.0 {
                    child.total_value / child.visits
                } else {
                    f64::NAN
                };
                (child.action.unwrap_or(-1), q, child.visits as u32, child.player_role)
            })
            .collect()
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

        // `expand_leaf` keeps children and priors the same length.  Reading
        // through `get` anyway means a future caller that breaks that contract
        // gets a badly-explored node, not a panic on a rayon worker that takes
        // the whole select_batch down with it.
        debug_assert_eq!(node.children.len(), node.priors.len());
        for i in 0..n_opts {
            let child_idx = node.children[i];
            let prior = node.priors.get(i).copied().unwrap_or(0.0);
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

/// `player_role` for an observation: 0 = the root player moves, 1 = the other.
///
/// The engine's `yourIndex` is an **absolute seat**, while `player_role` is
/// relative to whoever owns the search — `expand_leaf` negates the network's
/// value exactly when it is 1, and `puct_score` minimises exactly when it is 1.
/// Assigning `yourIndex` straight into it is therefore only correct when the
/// root player happens to sit in seat 0; from seat 1 every node in the tree
/// gets the opposite role, so the search negates its own value at its own
/// nodes and maximises the opponent's — it plays to lose, in half of all games,
/// with no error anywhere.
///
/// Defaults to 0 (the root player) when the observation carries no
/// `yourIndex`, which matches the previous behaviour for a malformed
/// observation: treat it as our own node rather than silently flipping a sign.
fn role_of(current: &Option<serde_json::Value>, our_player_index: i32) -> u8 {
    match current
        .as_ref()
        .and_then(|c| c.get("yourIndex").and_then(|v| v.as_i64()))
    {
        Some(seat) => {
            if seat as i32 == our_player_index {
                0
            } else {
                1
            }
        }
        None => 0,
    }
}

// ── Expansion ─────────────────────────────────────────────────────────────

/// Apply NN priors and value to a leaf node, then backpropagate.
///
/// * `leaf_idx` — index of the leaf node (must match the last node on the
///   selection path from `select_leaf`).
/// * `priors` — P(s, a) for each legal option, should sum to ~1.0.
/// * `value` — V(s) ∈ [-1, 1] from the NN value head, **in the perspective of
///   the player to move at this leaf**.  This function re-orients it into the
///   root player's frame before backup; callers must not pre-negate.
///
///   The network cannot supply a root-relative value: every observation it
///   sees is egocentric (the featurizer indexes every zone as
///   `[your_index, 1 - your_index]`), so "this is the opponent's node" is not
///   expressible in its input — it is a relation to a search root the network
///   knows nothing about.  The tree owns `player_role`, so the tree owns the
///   flip.
///
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
    // Role of the player to move at this leaf: 0 = us (root player), 1 = them.
    let player_role = tree.nodes[leaf_idx].player_role;
    // Re-orient the NN value into the root player's frame.  `terminal_value`
    // is exempt: it is already computed against `tree.our_player_index`.
    let oriented = if player_role == 1 { -value } else { value };

    // ── Store priors on the leaf ─────────────────────────────────────────
    {
        let leaf = &tree.nodes[leaf_idx];
        if leaf.is_terminal {
            // Terminal: use the terminal value, don't expand children.
            let tv = leaf.terminal_value.unwrap_or(oriented);
            backpropagate(tree, tv);
            tree.iter_count += 1;
            return Ok(());
        }
    } // release immutable borrow

    // Collect info needed for child creation
    let n_options = tree.nodes[leaf_idx].n_options;

    // One child per *scored* option.  `select_leaf` walks children and reads
    // the prior at the same index, so the two vectors must be the same length;
    // they were not, and the shorter one was `priors`:
    //
    //   - The featurizer caps an option list at O_MAX (64), so a node with
    //     more options than that came back with 64 priors against n_options
    //     children — an out-of-bounds index at exactly 64.
    //   - Multi-select added a STOP column with no child behind it, making
    //     priors one *longer* instead.
    //
    // The evaluator now drops STOP, and taking the min here caps the tree at
    // what the network actually scored: options past O_MAX go unsearched,
    // which is where the policy stands on them anyway.
    let mut priors = priors;
    let n_children = n_options.min(priors.len());
    priors.truncate(n_children);

    // Create placeholder children (no search_id yet — lazy creation).
    // The child at index `i` corresponds to option `i`.
    let mut child_indices: Vec<usize> = Vec::with_capacity(n_children);
    for opt_idx in 0..n_children {
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
    backpropagate(tree, oriented);

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

    // Determine player role from the observation.  `yourIndex` is an absolute
    // seat, `player_role` is relative to the root player — see `role_of`.
    let player_role = role_of(&obs.current, tree.our_player_index);

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

    let root_player_role = role_of(&root_obs.current, our_player_index);

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
    /// The engine that owns every `search_id` in this tree.
    ///
    /// A `search_id` is meaningful only to the agent that minted it:
    /// `SearchStep` and `SearchRelease` both take the agent pointer, and
    /// libcg looks the id up in *that* agent's table.  Binding the engine
    /// once, at `add_tree`, is what keeps the root — created before the
    /// tree has an id — on the same agent as every child and every
    /// release.  Deriving it as `tree_id % n_engines` at each call site
    /// did not: the root was minted on engine 0 while `realise_leaves`,
    /// `all_results` and `reset` addressed engine `tid % n`, so for every
    /// tree with `tid % n != 0` the child steps failed and the root state
    /// was never freed.  libcg exports no `AgentEnd`, so that leak is
    /// permanent — ~22 KB per root, measured.
    engine_idx: usize,
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
    /// Size of the engine pool, used to round-robin new trees across it.
    /// Each tree then keeps its own `engine_idx` — see [`ForestTree`].
    n_engines: usize,
}

impl PuctForest {
    pub fn new(n_engines: usize) -> Self {
        PuctForest {
            trees: Vec::new(),
            n_engines: if n_engines > 0 { n_engines } else { 1 },
        }
    }

    /// The engine the next [`add_tree`](Self::add_tree) will bind to.
    ///
    /// The caller must mint the root's `search_id` on this engine, because
    /// the root is created before the tree exists.
    pub fn next_engine_idx(&self) -> usize {
        self.trees.len() % self.n_engines
    }

    /// Add a new tree to the forest.  Returns the tree_id.
    ///
    /// `engine_idx` must be the engine that minted `root_node.search_id` —
    /// pass what [`next_engine_idx`](Self::next_engine_idx) returned.
    pub fn add_tree(
        &mut self,
        root_node: PuctNode,
        our_player_index: i32,
        config: PuctConfig,
        engine_idx: usize,
    ) -> usize {
        let id = self.trees.len();
        self.trees.push(ForestTree {
            tree: PuctTree::new(root_node, our_player_index),
            selection_path: Vec::new(),
            last_leaf: 0,
            iter_count: 0,
            config,
            engine_idx,
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

            let engine = &engines[ft.engine_idx % engines.len()];
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
    ///
    /// Takes `&mut self` because it releases each node's engine state and
    /// zeroes the id afterwards.  That zeroing is what makes [`reset`]
    /// safe to call on a forest whose results were already collected —
    /// `SearchRelease` on an id the engine has already freed is not a
    /// no-op on libcg's side.
    pub fn all_results(&mut self, engines: &[crate::engine::Engine]) -> Vec<ForestTreeResult> {
        self.trees
            .iter_mut()
            .enumerate()
            .map(|(tid, ft)| {
                // Release engine states for this tree, on the agent that
                // minted them — see `ForestTree::engine_idx`.
                let engine = &engines[ft.engine_idx % engines.len()];
                for node in &mut ft.tree.nodes {
                    if node.search_id != 0 {
                        engine.search_release(node.search_id);
                        node.search_id = 0;
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

    /// Drop every tree, keeping the engines alive.
    ///
    /// This is the only way to reuse a forest across search batches, and
    /// reuse is not an optimisation — it is required for the process to
    /// survive.  `AgentStart` has no counterpart in libcg's ABI (there is
    /// no `AgentEnd`), and `SearchEnd` only returns the arena to *that*
    /// agent for reuse, so an `Engine` that is dropped strands everything
    /// it ever allocated.  Building a fresh pool per search batch
    /// therefore leaks the whole arena, once per batch.
    ///
    /// Releases any node state [`all_results`] did not already release —
    /// the search loop can break early, and those ids belong to the
    /// long-lived agents now, not to a pool about to be dropped.
    pub fn reset(&mut self, engines: &[crate::engine::Engine]) -> usize {
        let n = self.trees.len();
        for ft in self.trees.iter_mut() {
            let engine = &engines[ft.engine_idx % engines.len()];
            for node in &mut ft.tree.nodes {
                if node.search_id != 0 {
                    engine.search_release(node.search_id);
                    node.search_id = 0;
                }
            }
        }
        self.trees.clear();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n_options: usize) -> PuctNode {
        PuctNode {
            search_id: 1,
            action: None,
            n_options,
            visits: 0.0,
            total_value: 0.0,
            priors: Vec::new(),
            children: Vec::new(),
            obs_json: String::new(),
            player_role: 0,
            is_terminal: false,
            terminal_value: None,
        }
    }

    fn tree_with_root(n_options: usize) -> PuctTree {
        let mut t = PuctTree::new(node(n_options), 0);
        t.selection_path = vec![0];
        t
    }

    /// The panic the user hit: the featurizer caps an option list at O_MAX, so
    /// a node with more engine options than that came back with fewer priors
    /// than the tree had children, and `select_leaf` indexed past the end.
    #[test]
    fn test_fewer_priors_than_options_truncates_the_tree() {
        let mut tree = tree_with_root(70);
        let priors = vec![1.0 / 64.0; 64];
        expand_leaf(&mut tree, 0, priors, 0.5, "{}".to_string()).unwrap();

        assert_eq!(tree.nodes[0].children.len(), 64, "tree must match the priors");
        assert_eq!(tree.nodes[0].priors.len(), 64);

        // The traversal that used to panic.
        let cfg = PuctConfig::default();
        tree.selection_path.clear();
        let leaf = select_leaf(&mut tree, &cfg).expect("selection must not fail");
        assert!(leaf < tree.nodes.len());
    }

    /// Multi-select adds a STOP column with no child behind it, which made
    /// priors one *longer* than children.
    #[test]
    fn test_more_priors_than_options_are_dropped() {
        let mut tree = tree_with_root(3);
        let priors = vec![0.25, 0.25, 0.25, 0.25]; // 3 options + STOP
        expand_leaf(&mut tree, 0, priors, 0.0, "{}".to_string()).unwrap();

        assert_eq!(tree.nodes[0].children.len(), 3);
        assert_eq!(tree.nodes[0].priors.len(), 3, "the STOP prior has no child");
    }

    /// The ordinary case must be untouched.
    #[test]
    fn test_matching_lengths_are_preserved() {
        let mut tree = tree_with_root(4);
        expand_leaf(&mut tree, 0, vec![0.1, 0.2, 0.3, 0.4], 0.0, "{}".to_string()).unwrap();
        assert_eq!(tree.nodes[0].children.len(), 4);
        assert_eq!(tree.nodes[0].priors, vec![0.1, 0.2, 0.3, 0.4]);
    }
}

#[cfg(test)]
mod role_of_tests {
    use super::role_of;
    use serde_json::json;

    fn obs(seat: i64) -> Option<serde_json::Value> {
        Some(json!({ "yourIndex": seat }))
    }

    /// From seat 0 the old code was accidentally right, so this passes either
    /// way — it is here to pin that the fix did not break the common case.
    #[test]
    fn seat_zero_root_sees_itself_as_the_max_player() {
        assert_eq!(role_of(&obs(0), 0), 0);
        assert_eq!(role_of(&obs(1), 0), 1);
    }

    /// The bug: `player_role` was the raw seat, so a root player in seat 1 got
    /// role 1 at its *own* nodes — `expand_leaf` then negated its own value and
    /// `puct_score` minimised it, i.e. the search played to lose.
    #[test]
    fn seat_one_root_is_still_the_max_player() {
        assert_eq!(role_of(&obs(1), 1), 0, "our own node must be a max node");
        assert_eq!(role_of(&obs(0), 1), 1, "their node must be a min node");
    }

    /// Role is a relation between two seats, so it must be symmetric under
    /// swapping which seat we occupy.
    #[test]
    fn role_depends_on_both_seats_not_just_the_observation() {
        for seat in 0..2i64 {
            for ours in 0..2i32 {
                let expected = if seat as i32 == ours { 0 } else { 1 };
                assert_eq!(role_of(&obs(seat), ours), expected, "seat={seat} ours={ours}");
            }
        }
    }

    #[test]
    fn a_missing_your_index_defaults_to_our_own_node() {
        assert_eq!(role_of(&None, 1), 0);
        assert_eq!(role_of(&Some(json!({})), 1), 0);
    }
}
