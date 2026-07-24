/// UCT (Upper Confidence Bound for Trees) / MCTS search over the engine's
/// determinized forward model.
///
/// Each node corresponds to an engine `search_id`.  The tree is stored in a
/// flat `Vec<MctsNode>`.  Rollout uses random action selection driven by
/// `search_step`; intermediate rollout states are released immediately.

use rand::Rng;

use crate::engine::{Engine, SearchObservation, SearchResult};

// ── Configuration ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of MCTS iterations (selection → expansion → simulation → backprop).
    pub iterations: u32,
    /// UCB1 exploration constant.  √2 ≈ 1.414.
    pub ucb_c: f64,
    /// Maximum rollout depth before truncation (safety limit).
    pub max_rollout_depth: u32,
    /// Random seed for reproducibility.
    #[allow(dead_code)]
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            iterations: 300,
            ucb_c: 1.414,
            max_rollout_depth: 200,
            seed: 42,
        }
    }
}

// ── Tree nodes ────────────────────────────────────────────────────────────

struct MctsNode {
    /// Engine search state id.
    search_id: i64,
    /// Which option was chosen to arrive at this node (None for root).
    action: Option<i32>,
    /// Number of legal options at this node.
    #[allow(dead_code)]
    n_options: usize,
    /// Number of times this node was visited.
    visits: f64,
    /// Sum of backpropagated values (+1 win / 0 draw / -1 loss, from our view).
    total_value: f64,
    /// Indices of expanded children (in `MctsTree::nodes`).
    children: Vec<usize>,
    /// Option indices not yet tried as children.
    untried_actions: Vec<i32>,
    /// Whether this state is terminal (game over).
    is_terminal: bool,
    /// If terminal, the result value (from our perspective).
    terminal_value: Option<f64>,
}

struct MctsTree {
    nodes: Vec<MctsNode>,
}

impl MctsTree {
    fn new() -> Self {
        MctsTree { nodes: Vec::new() }
    }

    fn add_node(&mut self, node: MctsNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    fn best_child(&self, node_idx: usize, c: f64) -> Option<usize> {
        let parent = &self.nodes[node_idx];
        if parent.children.is_empty() {
            return None;
        }
        let total_visits = parent.visits;
        let mut best_idx = parent.children[0];
        let mut best_score = ucb1(&self.nodes[best_idx], total_visits, c);

        for &child_idx in &parent.children[1..] {
            let score = ucb1(&self.nodes[child_idx], total_visits, c);
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }
        Some(best_idx)
    }

    fn all_search_ids(&self) -> Vec<i64> {
        self.nodes.iter().map(|n| n.search_id).collect()
    }
}

fn ucb1(child: &MctsNode, total_parent_visits: f64, c: f64) -> f64 {
    if child.visits == 0.0 {
        return f64::INFINITY;
    }
    let exploit = child.total_value / child.visits;
    let explore = c * (total_parent_visits.ln() / child.visits).sqrt();
    exploit + explore
}

// ── Public result type ────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SearchPlan {
    /// Chosen option indices.
    pub indices: Vec<i32>,
    /// Per-option visit counts for diagnostics.
    pub visit_counts: Vec<(i32, u32)>,
    /// Total iterations performed.
    pub iterations: u32,
    /// Number of tree nodes created.
    pub nodes_created: usize,
}

// ── Public entry point ────────────────────────────────────────────────────

pub fn search(
    engine: &Engine,
    root: &SearchResult,
    config: &MctsConfig,
    rng: &mut impl Rng,
    our_player_index: i32,
) -> Result<SearchPlan, String> {
    let root_obs: SearchObservation = serde_json::from_str(&root.observation_json)
        .map_err(|e| format!("root obs parse: {e}"))?;

    let select = root_obs
        .select
        .as_ref()
        .ok_or("root observation has no select")?;

    let max_count = select.max_count.max(1) as usize;
    let n_options = select.option.len();

    if n_options == 0 {
        return Ok(SearchPlan {
            indices: vec![],
            visit_counts: vec![],
            iterations: 0,
            nodes_created: 0,
        });
    }

    if max_count == 1 {
        single_select_mcts(engine, root, n_options, config, rng, our_player_index)
    } else {
        multi_select_greedy(engine, root, n_options, max_count, config, rng, our_player_index)
    }
}

// ── Single-select MCTS ────────────────────────────────────────────────────

fn single_select_mcts(
    engine: &Engine,
    root: &SearchResult,
    n_options: usize,
    config: &MctsConfig,
    rng: &mut impl Rng,
    our_player_index: i32,
) -> Result<SearchPlan, String> {
    let root_is_terminal = is_terminal_json_obs(&root.observation_json);
    let root_tv = if root_is_terminal {
        Some(terminal_value_from_obs_json(&root.observation_json, our_player_index))
    } else {
        None
    };

    let mut tree = MctsTree::new();
    let root_idx = tree.add_node(MctsNode {
        search_id: root.search_id,
        action: None,
        n_options,
        visits: 1.0,
        total_value: 0.0,
        children: Vec::with_capacity(n_options),
        untried_actions: (0..n_options as i32).collect(),
        is_terminal: root_is_terminal,
        terminal_value: root_tv,
    });

    for _iter in 0..config.iterations {
        // 1. Selection — walk down tree via UCB1 until leaf or unexpanded
        let mut path: Vec<usize> = vec![root_idx];
        let mut current = root_idx;

        loop {
            let node = &tree.nodes[current];
            if node.is_terminal || !node.untried_actions.is_empty() {
                break;
            }
            match tree.best_child(current, config.ucb_c) {
                Some(child_idx) => {
                    path.push(child_idx);
                    current = child_idx;
                }
                None => break,
            }
        }

        let leaf_idx = current;

        // 2. Expansion — pick one untried action
        let child_idx = if !tree.nodes[leaf_idx].is_terminal
            && !tree.nodes[leaf_idx].untried_actions.is_empty()
        {
            let untried = &tree.nodes[leaf_idx].untried_actions;
            let pick_i = rng.gen_range(0..untried.len());
            let action_idx = untried[pick_i];

            // Remove from untried
            tree.nodes[leaf_idx].untried_actions.remove(pick_i);

            // Call engine
            match engine.search_step(tree.nodes[leaf_idx].search_id, &[action_idx]) {
                Ok(child_result) => {
                    let child_is_term = is_terminal_json_obs(&child_result.observation_json);
                    let child_tv = if child_is_term {
                        Some(terminal_value_from_obs_json(
                            &child_result.observation_json,
                            our_player_index,
                        ))
                    } else {
                        None
                    };
                    let child_n_opts = count_options_from_json(&child_result.observation_json);

                    let child = MctsNode {
                        search_id: child_result.search_id,
                        action: Some(action_idx),
                        n_options: child_n_opts,
                        visits: 0.0,
                        total_value: 0.0,
                        children: Vec::with_capacity(child_n_opts),
                        untried_actions: if child_is_term {
                            Vec::new()
                        } else {
                            (0..child_n_opts as i32).collect()
                        },
                        is_terminal: child_is_term,
                        terminal_value: child_tv,
                    };
                    let idx = tree.add_node(child);
                    tree.nodes[leaf_idx].children.push(idx);
                    path.push(idx);
                    Some(idx)
                }
                Err(_) => {
                    // Engine rejected the action — skip
                    None
                }
            }
        } else {
            None
        };

        // 3. Simulation (rollout)
        let value = match child_idx {
            Some(c_idx) => {
                let child = &tree.nodes[c_idx];
                if child.is_terminal {
                    child.terminal_value.unwrap_or(0.0)
                } else {
                    // Rollout needs SearchResult with observation_json.  We don't
                    // have it cached for tree nodes, so re-derive by calling
                    // search_step with the first untried action, getting the
                    // observation, then rolling out from there.
                    //
                    // Better approach: start rollout from parent's search_id with
                    // the chosen action, and continue from there.
                    rollout_from_action(
                        engine,
                        tree.nodes[leaf_idx].search_id,
                        action_idx_from_node(&tree.nodes[c_idx]),
                        config.max_rollout_depth,
                        rng,
                        our_player_index,
                    )
                    .unwrap_or(0.0)
                }
            }
            None => {
                // Leaf is terminal
                tree.nodes[leaf_idx].terminal_value.unwrap_or(0.0)
            }
        };

        // 4. Backpropagation
        for &node_idx in &path {
            tree.nodes[node_idx].visits += 1.0;
            tree.nodes[node_idx].total_value += value;
        }
        if let Some(c_idx) = child_idx {
            tree.nodes[c_idx].visits += 1.0;
            tree.nodes[c_idx].total_value += value;
        }
    }

    // ── Choose best action ────────────────────────────────────────────────
    let root_node = &tree.nodes[root_idx];
    let mut visit_counts: Vec<(i32, u32)> = root_node
        .children
        .iter()
        .map(|&c| {
            let child = &tree.nodes[c];
            (child.action.unwrap_or(-1), child.visits as u32)
        })
        .collect();
    visit_counts.sort_by_key(|(_, v)| std::cmp::Reverse(*v));

    let best_action = visit_counts.first().map(|(a, _)| *a).unwrap_or(0);

    // Cleanup engine states
    for sid in tree.all_search_ids() {
        engine.search_release(sid);
    }

    Ok(SearchPlan {
        indices: vec![best_action],
        visit_counts,
        iterations: config.iterations,
        nodes_created: tree.nodes.len(),
    })
}

fn action_idx_from_node(node: &MctsNode) -> i32 {
    node.action.unwrap_or(0)
}

// ── Rollout (random playout using the engine) ──────────────────────────────

/// Start a rollout by first stepping to `action`, then continuing with random
/// actions until terminal or depth limit.  All engine states created during
/// rollout are released.
fn rollout_from_action(
    engine: &Engine,
    parent_search_id: i64,
    action: i32,
    max_depth: u32,
    rng: &mut impl Rng,
    our_player_index: i32,
) -> Result<f64, String> {
    // Step into the action
    let first = engine
        .search_step(parent_search_id, &[action])
        .map_err(|e| format!("rollout first step: {e}"))?;

    rollout_from_result(engine, first, max_depth, rng, our_player_index)
}

/// Continue a random rollout from `current` (which we own — its search_id will
/// be released by us).
fn rollout_from_result(
    engine: &Engine,
    current: SearchResult,
    max_depth: u32,
    rng: &mut impl Rng,
    our_player_index: i32,
) -> Result<f64, String> {
    let mut cur = current;

    for _depth in 0..max_depth {
        // Check terminal
        if is_terminal_json_obs(&cur.observation_json) {
            let val = terminal_value_from_obs_json(&cur.observation_json, our_player_index);
            engine.search_release(cur.search_id);
            return Ok(val);
        }

        let n_opts = count_options_from_json(&cur.observation_json);
        if n_opts == 0 {
            engine.search_release(cur.search_id);
            return Ok(0.0);
        }

        let opt = rng.gen_range(0..n_opts) as i32;

        match engine.search_step(cur.search_id, &[opt]) {
            Ok(next) => {
                engine.search_release(cur.search_id);
                cur = next;
            }
            Err(_) => {
                engine.search_release(cur.search_id);
                return Ok(0.0);
            }
        }
    }

    // Truncated
    engine.search_release(cur.search_id);
    Ok(0.0)
}

// ── Multi-select greedy (v1 simplified) ───────────────────────────────────

fn multi_select_greedy(
    engine: &Engine,
    root: &SearchResult,
    _n_options: usize,
    max_count: usize,
    config: &MctsConfig,
    rng: &mut impl Rng,
    our_player_index: i32,
) -> Result<SearchPlan, String> {
    let mut chosen: Vec<i32> = Vec::new();
    let mut current_search_id = root.search_id;

    for pick_n in 0..max_count {
        let n_opts = count_options_via_engine(engine, current_search_id)?;
        if n_opts == 0 {
            break;
        }

        // Try each option with a short rollout
        let mut scores: Vec<(i32, f64)> = Vec::new();
        for opt_idx in 0..n_opts.min(64) as i32 {
            if chosen.contains(&opt_idx) {
                continue;
            }
            match engine.search_step(current_search_id, &[opt_idx]) {
                Ok(child) => {
                    let val = rollout_from_result(
                        engine,
                        child,
                        config.max_rollout_depth / 2,
                        rng,
                        our_player_index,
                    )
                    .unwrap_or(0.0);
                    scores.push((opt_idx, val));
                }
                Err(_) => continue,
            }
        }

        if scores.is_empty() {
            break;
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = scores[0].0;
        chosen.push(best);

        // Step forward
        match engine.search_step(current_search_id, &[best]) {
            Ok(next) => {
                if pick_n > 0 {
                    engine.search_release(current_search_id);
                }
                current_search_id = next.search_id;
            }
            Err(_) => break,
        }
    }

    engine.search_release(current_search_id);

    Ok(SearchPlan {
        indices: chosen,
        visit_counts: vec![],
        iterations: config.iterations,
        nodes_created: 0,
    })
}

/// Try to get n_options from a search_id by calling search_step with the first
/// option, reading the observation, then releasing.  This is a hack — a better
/// design would cache observations alongside search_ids.
fn count_options_via_engine(engine: &Engine, search_id: i64) -> Result<usize, String> {
    // Call search_step with a dummy action [0]; if it succeeds, read n_options
    // and release.  If there are 0 options this will fail, which is fine.
    match engine.search_step(search_id, &[0]) {
        Ok(result) => {
            let n = count_options_from_json(&result.observation_json);
            engine.search_release(result.search_id);
            Ok(n)
        }
        Err(_) => Ok(0),
    }
}

// ── JSON helpers (work on &str or &SearchObservation) ──────────────────────

fn is_terminal_json_obs(obs_json: &str) -> bool {
    let obs: Option<SearchObservation> = serde_json::from_str(obs_json).ok();
    obs.and_then(|o| o.current)
        .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
        .map(|r| r != -1)
        .unwrap_or(false)
}

fn terminal_value_from_obs_json(obs_json: &str, our_idx: i32) -> f64 {
    let obs: Option<SearchObservation> = serde_json::from_str(obs_json).ok();
    let result = obs
        .and_then(|o| o.current)
        .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
        .unwrap_or(-1);

    match result {
        0 => {
            if our_idx == 0 {
                1.0
            } else {
                -1.0
            }
        }
        1 => {
            if our_idx == 1 {
                1.0
            } else {
                -1.0
            }
        }
        2 => 0.0, // draw
        _ => 0.0,
    }
}

fn count_options_from_json(obs_json: &str) -> usize {
    let obs: Option<SearchObservation> = serde_json::from_str(obs_json).ok();
    obs.and_then(|o| o.select)
        .map(|s| s.option.len())
        .unwrap_or(0)
}
