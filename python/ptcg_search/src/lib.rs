//! ptcg_search — MCTS search planner for Pokémon TCG AI Battle.
//!
//! This crate links against `libcg.so` (the game engine) at runtime and
//! exposes a single C-ABI function `search_plan` for Python to call via
//! `ctypes`.

mod bridge;
mod engine;
mod ffi;
mod guessing;
mod mcts;
mod puct;
mod vec_env;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

use rand::rngs::StdRng;
use rand::SeedableRng;

use engine::Engine;
use guessing::build_guesses;
use mcts::MctsConfig;

// ── C FFI entry point ──────────────────────────────────────────────────────

/// Run MCTS search for the best action at a decision point.
///
/// # Parameters (all C strings / ints)
///
/// * `obs_json` — Raw observation dict as JSON string (from `agent(obs_dict)`).
/// * `lib_path` — Filesystem path to `libcg.so`.
/// * `fixed_deck_json` — Our FIXED_DECK as a JSON array of 60 ints.
/// * `opp_deck_json` — Opponent's guessed deck as a JSON array of ints (can be
///   empty array `[]` to fall back to mirror).
/// * `iterations` — MCTS iteration budget (e.g. 300).
/// * `seed` — Random seed for reproducibility.
/// * `host_initialized` — Non-zero if the calling process has **already** called
///   `GameInitialize` on this libcg.so (every Python caller has, via `cg.sim`'s
///   import).  Getting this wrong is fatal, not recoverable: a second
///   `GameInitialize` throws a C++ exception across the FFI boundary and aborts
///   the process.  See [`Engine::load`].
///
/// # Returns
///
/// A JSON string (caller must free with `search_plan_free`):
/// ```json
/// {"indices": [0], "visit_counts": [[0, 210], [3, 90]], "root_value": 0.31,
///  "iterations": 300, "nodes_created": 42, "error": null}
/// ```
///
/// `visit_counts` is `[[option_index, visits], ...]` sorted by descending visits,
/// and `root_value` is the mean playout outcome at the root in `[-1, 1]`.  Both
/// feed RL_SPEC §8.3's search-distillation targets.  `root_value` is `null` when
/// no tree was built (multi-select decisions, or an empty option list) — that is
/// distinct from a value of `0.0`, and a consumer must not conflate them.
///
/// On error, `"indices"` is `[]` and `"error"` contains the message.
///
/// # Safety
///
/// All string arguments must be valid UTF-8, null-terminated C strings.
/// The returned pointer must be freed with `search_plan_free`.
#[no_mangle]
pub unsafe extern "C" fn search_plan(
    obs_json: *const c_char,
    lib_path: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    seed: c_int,
    host_initialized: c_int,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        search_plan_impl(
            obs_json,
            lib_path,
            fixed_deck_json,
            opp_deck_json,
            iterations,
            seed,
            host_initialized,
        )
    }));

    match result {
        Ok(json) => {
            CString::new(json).unwrap_or_else(|_| CString::new("{}").unwrap()).into_raw()
        }
        Err(_) => {
            let err = r#"{"indices":[],"visit_counts":[],"root_value":null,"iterations":0,"nodes_created":0,"error":"panic in search_plan"}"#;
            CString::new(err).unwrap().into_raw()
        }
    }
}

fn search_plan_impl(
    obs_json: *const c_char,
    lib_path: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    seed: c_int,
    host_initialized: c_int,
) -> String {
    let obs_str = unsafe { cstr_to_str(obs_json) };
    let lib_str = unsafe { cstr_to_str(lib_path) };
    let fixed_str = unsafe { cstr_to_str(fixed_deck_json) };
    let opp_str = unsafe { cstr_to_str(opp_deck_json) };

    let obs: serde_json::Value = match serde_json::from_str(&obs_str) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("obs_json parse: {e}")),
    };
    let fixed_deck: Vec<i32> = match serde_json::from_str(&fixed_str) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("fixed_deck_json parse: {e}")),
    };
    let opp_deck_template: Vec<i32> = match serde_json::from_str(&opp_str) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("opp_deck_json parse: {e}")),
    };

    // Handle deck-selection step (select is None)
    if obs.get("select").and_then(|s| s.as_object()).is_none() {
        return serde_json::json!({
            "indices": fixed_deck,
            "visit_counts": [],
            "root_value": null,
            "iterations": 0,
            "nodes_created": 0,
            "error": null
        })
        .to_string();
    }

    // Load engine.  Only call GameInitialize if the host has not already done so —
    // a second call aborts the process (see Engine::load).
    let engine = match unsafe { Engine::load(&lib_str, host_initialized == 0) } {
        Ok(e) => e,
        Err(e) => return error_json(&format!("Engine::load: {e}")),
    };

    // Build hidden-state guesses
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let guesses = match build_guesses(&obs_str, &fixed_deck, &opp_deck_template, &mut rng) {
        Ok(g) => g,
        Err(e) => return error_json(&format!("build_guesses: {e}")),
    };

    // Get search_begin_input from observation.  Empty is fatal, not
    // recoverable — see puct_forest_add_root_impl.
    let sbi = obs
        .get("search_begin_input")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if sbi.is_empty() {
        return error_json("empty search_begin_input");
    }

    // Start search
    let root = match engine.search_begin(
        sbi,
        &guesses.your_deck,
        &guesses.your_prize,
        &guesses.opponent_deck,
        &guesses.opponent_prize,
        &guesses.opponent_hand,
        &guesses.opponent_active,
        false, // manual_coin
    ) {
        Ok(r) => r,
        Err(e) => return error_json(&format!("search_begin: {e}")),
    };

    // Determine which player we are
    let our_player_index = obs
        .get("current")
        .and_then(|c| c.get("yourIndex"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    // Run MCTS
    let config = MctsConfig {
        iterations: iterations.max(1) as u32,
        seed: seed as u64,
        ..MctsConfig::default()
    };

    match mcts::search(&engine, &root, &config, &mut rng, our_player_index) {
        Ok(plan) => {
            serde_json::json!({
                "indices": plan.indices,
                "visit_counts": plan.visit_counts,
                "root_value": plan.root_value,
                "iterations": plan.iterations,
                "nodes_created": plan.nodes_created,
                "error": null
            })
            .to_string()
        }
        Err(e) => error_json(&format!("mcts::search: {e}")),
    }
}

/// Free a string previously returned by `search_plan`.
///
/// # Safety
///
/// `ptr` must have been returned by `search_plan` and not freed before.
#[no_mangle]
pub unsafe extern "C" fn search_plan_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

unsafe fn cstr_to_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({
        "indices": [],
        "visit_counts": [],
        "root_value": null,
        "iterations": 0,
        "nodes_created": 0,
        "error": msg
    })
    .to_string()
}

// ── PUCT C-ABI (Python-driven split: select → evaluate → expand) ───────────

use puct::{PuctConfig, PuctNode, PuctTree};

/// Opaque handle for a PUCT search session.
struct PuctSearchHandle {
    engine: Engine,
    tree: PuctTree,
    config: PuctConfig,
    /// Path from the last `puct_select`: node indices from root to leaf.
    /// Consumed by `puct_expand` for backpropagation.
    selection_path: Vec<usize>,
    /// The leaf index from the last `puct_select`.
    last_leaf: usize,
}

/// Create a PUCT search tree with determinization.
///
/// Returns an opaque handle (as i64) that must be freed with `puct_free`.
///
/// # Safety
///
/// All string arguments must be valid UTF-8, null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn puct_init(
    obs_json: *const c_char,
    lib_path: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    c_puct: f64,
    seed: c_int,
    host_initialized: c_int,
) -> i64 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_init_impl(obs_json, lib_path, fixed_deck_json, opp_deck_json, iterations, c_puct, seed, host_initialized)
    }));
    match result {
        Ok(Ok(handle)) => Box::into_raw(Box::new(handle)) as i64,
        Ok(Err(_)) | Err(_) => 0, // error: null handle
    }
}

fn puct_init_impl(
    obs_json: *const c_char,
    lib_path: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    c_puct: f64,
    seed: c_int,
    host_initialized: c_int,
) -> Result<PuctSearchHandle, String> {
    let obs_str = unsafe { cstr_to_str(obs_json) };
    let lib_str = unsafe { cstr_to_str(lib_path) };
    let fixed_str = unsafe { cstr_to_str(fixed_deck_json) };
    let opp_str = unsafe { cstr_to_str(opp_deck_json) };

    let obs: serde_json::Value = serde_json::from_str(&obs_str)
        .map_err(|e| format!("obs_json parse: {e}"))?;
    let fixed_deck: Vec<i32> = serde_json::from_str(&fixed_str)
        .map_err(|e| format!("fixed_deck_json parse: {e}"))?;
    let opp_deck_template: Vec<i32> = serde_json::from_str(&opp_str)
        .map_err(|e| format!("opp_deck_json parse: {e}"))?;

    // Handle deck-selection step
    if obs.get("select").and_then(|s| s.as_object()).is_none() {
        return Err("deck selection step: no select in observation".into());
    }

    // Load engine
    let engine = unsafe { Engine::load(&lib_str, host_initialized == 0) }
        .map_err(|e| format!("Engine::load: {e}"))?;

    // Build hidden-state guesses
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let guesses = guessing::build_guesses(&obs_str, &fixed_deck, &opp_deck_template, &mut rng)
        .map_err(|e| format!("build_guesses: {e}"))?;

    let sbi = obs
        .get("search_begin_input")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if sbi.is_empty() {
        return Err("empty search_begin_input".into());
    }

    let root = engine
        .search_begin(
            sbi,
            &guesses.your_deck,
            &guesses.your_prize,
            &guesses.opponent_deck,
            &guesses.opponent_prize,
            &guesses.opponent_hand,
            &guesses.opponent_active,
            false,
        )
        .map_err(|e| format!("search_begin: {e}"))?;

    let our_player_index = obs
        .get("current")
        .and_then(|c| c.get("yourIndex"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let root_obs: engine::SearchObservation =
        serde_json::from_str(&root.observation_json)
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

    let player_role = root_obs
        .current
        .as_ref()
        .and_then(|c| c.get("yourIndex").and_then(|v| v.as_i64()))
        .map(|v| v as u8)
        .unwrap_or(0);

    let root_node = PuctNode {
        search_id: root.search_id,
        action: None,
        n_options,
        visits: 1.0,
        total_value: 0.0,
        priors: Vec::new(),
        children: Vec::new(),
        obs_json: root.observation_json.clone(),
        player_role,
        is_terminal,
        terminal_value: None,
    };

    let tree = PuctTree::new(root_node, our_player_index);
    let config = PuctConfig {
        iterations: iterations.max(1) as u32,
        c_puct: if c_puct > 0.0 { c_puct } else { 2.0 },
        seed: seed as u64,
        ..PuctConfig::default()
    };

    Ok(PuctSearchHandle {
        engine,
        tree,
        config,
        selection_path: Vec::new(),
        last_leaf: 0,
    })
}

/// Walk tree from root to a leaf that needs NN evaluation.
///
/// Returns a JSON string (caller must free with `puct_free_result`):
/// ```json
/// {
///   "leaf_obs_json": "...",
///   "player_role": 0,
///   "is_terminal": false,
///   "n_options": 10,
///   "tree_done": false,
///   "iter_count": 5,
///   "error": null
/// }
/// ```
///
/// `tree_done: true` means all iterations are complete or the root is terminal.
/// The caller should then call `puct_result` to get the final visit counts.
///
/// # Safety
///
/// `handle` must be a valid pointer returned by `puct_init`.
#[no_mangle]
pub unsafe extern "C" fn puct_select(handle: i64) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_select_impl(handle)
    }));
    match result {
        Ok(json) => CString::new(json).unwrap_or_else(|_| CString::new("{}").unwrap()).into_raw(),
        Err(_) => {
            let err = r#"{"leaf_obs_json":"","player_role":0,"is_terminal":false,"n_options":0,"tree_done":true,"iter_count":0,"error":"panic in puct_select"}"#;
            CString::new(err).unwrap().into_raw()
        }
    }
}

fn puct_select_impl(handle: i64) -> String {
    if handle == 0 {
        return r#"{"leaf_obs_json":"","player_role":0,"is_terminal":false,"n_options":0,"tree_done":true,"iter_count":0,"error":"null handle"}"#.to_string();
    }
    let h = unsafe { &mut *(handle as *mut PuctSearchHandle) };

    // Check if done
    if h.tree.iter_count >= h.config.iterations {
        return serde_json::json!({
            "leaf_obs_json": "",
            "player_role": 0,
            "is_terminal": false,
            "n_options": 0,
            "tree_done": true,
            "iter_count": h.tree.iter_count,
            "error": null
        }).to_string();
    }

    // The root is terminal — nothing to search
    if h.tree.nodes[0].is_terminal {
        return serde_json::json!({
            "leaf_obs_json": "",
            "player_role": 0,
            "is_terminal": true,
            "n_options": 0,
            "tree_done": true,
            "iter_count": h.tree.iter_count,
            "error": null
        }).to_string();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(
        h.config.seed.wrapping_add(h.tree.iter_count as u64)
    );

    // Select leaf
    match puct::select_leaf(&mut h.tree, &h.config) {
        Ok(leaf_idx) => {
            let leaf = &h.tree.nodes[leaf_idx];

            // If this is a placeholder child (search_id == 0), realise it first
            let obs_json = if leaf.search_id == 0 && leaf.action.is_some() {
                // Find parent: the node before this leaf on the selection path
                let parent_idx = if h.tree.selection_path.len() >= 2 {
                    h.tree.selection_path[h.tree.selection_path.len() - 2]
                } else {
                    0 // root is parent
                };
                match puct::realise_child(&mut h.tree, parent_idx, leaf_idx, &h.engine) {
                    Ok(json) => json,
                    Err(e) => {
                        return serde_json::json!({
                            "leaf_obs_json": "",
                            "player_role": 0,
                            "is_terminal": false,
                            "n_options": 0,
                            "tree_done": true,
                            "iter_count": h.tree.iter_count,
                            "error": format!("realise_child: {e}")
                        }).to_string();
                    }
                }
            } else {
                leaf.obs_json.clone()
            };

            let leaf_ref = &h.tree.nodes[leaf_idx];

            h.selection_path = h.tree.selection_path.clone();
            h.last_leaf = leaf_idx;

            serde_json::json!({
                "leaf_obs_json": obs_json,
                "player_role": leaf_ref.player_role,
                "is_terminal": leaf_ref.is_terminal,
                "n_options": leaf_ref.n_options,
                "tree_done": false,
                "iter_count": h.tree.iter_count,
                "error": null
            }).to_string()
        }
        Err(e) => {
            serde_json::json!({
                "leaf_obs_json": "",
                "player_role": 0,
                "is_terminal": false,
                "n_options": 0,
                "tree_done": true,
                "iter_count": h.tree.iter_count,
                "error": format!("select_leaf: {e}")
            }).to_string()
        }
    }
}

/// Expand a leaf with NN priors and value, then backpropagate.
///
/// `priors_json` is a JSON array of floats, one per legal option.
/// `value` is the NN value head output in [-1, 1].
///
/// Returns 0 on success, non-zero on error.
///
/// # Safety
///
/// `handle` must be a valid pointer returned by `puct_init`.
/// `priors_json` must be a valid null-terminated UTF-8 JSON string.
#[no_mangle]
pub unsafe extern "C" fn puct_expand(
    handle: i64,
    priors_json: *const c_char,
    value: f64,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_expand_impl(handle, priors_json, value)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(_)) | Err(_) => -1,
    }
}

fn puct_expand_impl(
    handle: i64,
    priors_json: *const c_char,
    value: f64,
) -> Result<(), String> {
    if handle == 0 {
        return Err("null handle".into());
    }
    let h = unsafe { &mut *(handle as *mut PuctSearchHandle) };

    let priors_str = unsafe { cstr_to_str(priors_json) };
    let priors: Vec<f64> = serde_json::from_str(&priors_str)
        .map_err(|e| format!("priors_json parse: {e}"))?;

    let leaf_idx = h.last_leaf;
    let obs_json = h.tree.nodes[leaf_idx].obs_json.clone();

    // Restore the selection path
    h.tree.selection_path = h.selection_path.clone();

    puct::expand_leaf(&mut h.tree, leaf_idx, priors, value, obs_json)
}

/// Get the final result of the PUCT search.
///
/// Returns a JSON string (caller must free with `puct_free_result`):
/// ```json
/// {"visit_counts": [[0, 75], [3, 53]], "root_value": 0.31, "iterations": 128, "nodes_created": 384, "error": null}
/// ```
///
/// # Safety
///
/// `handle` must be a valid pointer returned by `puct_init`.
#[no_mangle]
pub unsafe extern "C" fn puct_result(handle: i64) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_result_impl(handle)
    }));
    match result {
        Ok(json) => CString::new(json).unwrap_or_else(|_| CString::new("{}").unwrap()).into_raw(),
        Err(_) => {
            let err = r#"{"visit_counts":[],"root_value":null,"iterations":0,"nodes_created":0,"error":"panic in puct_result"}"#;
            CString::new(err).unwrap().into_raw()
        }
    }
}

fn puct_result_impl(handle: i64) -> String {
    if handle == 0 {
        return r#"{"visit_counts":[],"root_value":null,"iterations":0,"nodes_created":0,"error":"null handle"}"#.to_string();
    }
    let h = unsafe { &mut *(handle as *mut PuctSearchHandle) };

    // Cleanup engine states
    for node in &h.tree.nodes {
        if node.search_id != 0 {
            h.engine.search_release(node.search_id);
        }
    }

    serde_json::json!({
        "visit_counts": h.tree.visit_counts(),
        "root_value": h.tree.root_value(),
        "iterations": h.tree.iter_count,
        "nodes_created": h.tree.nodes.len(),
        "error": null
    }).to_string()
}

/// Free a PUCT search handle and all associated engine resources.
///
/// # Safety
///
/// `handle` must be a valid pointer returned by `puct_init` and not freed before.
#[no_mangle]
pub unsafe extern "C" fn puct_free(handle: i64) {
    if handle != 0 {
        let _h = unsafe { Box::from_raw(handle as *mut PuctSearchHandle) };
        // Engine is dropped here, which calls SearchEnd
    }
}

/// Free a string returned by `puct_select` or `puct_result`.
///
/// # Safety
///
/// `ptr` must have been returned by `puct_select` or `puct_result` and not freed before.
#[no_mangle]
pub unsafe extern "C" fn puct_free_result(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

// ── PUCT Forest C-ABI (Phase 3b: batched multi-tree) ─────────────────────

use engine::EnginePool;
use puct::{ForestExpansion, PuctForest};

struct PuctForestHandle {
    pool: EnginePool,
    forest: PuctForest,
}

/// Create a PUCT forest with an engine pool.
///
/// `n_engines`: number of Engine instances (one per rayon worker).
/// Returns an opaque handle (i64), or 0 on error.
#[no_mangle]
pub unsafe extern "C" fn puct_forest_create(
    n_engines: c_int,
    lib_path: *const c_char,
    host_initialized: c_int,
) -> i64 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let lib_str = unsafe { cstr_to_str(lib_path) };
        let n = (n_engines.max(1)) as usize;
        let pool = EnginePool::new(&lib_str, n, host_initialized == 0)
            .map_err(|e| format!("EnginePool::new: {e}"))?;
        let n_eng = pool.len();
        Ok::<PuctForestHandle, String>(PuctForestHandle {
            pool,
            forest: PuctForest::new(n_eng),
        })
    }));
    match result {
        Ok(Ok(handle)) => Box::into_raw(Box::new(handle)) as i64,
        Ok(Err(_)) | Err(_) => 0,
    }
}

/// Add a root search state to the forest.
///
/// Returns the tree_id (≥ 0), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn puct_forest_add_root(
    handle: i64,
    obs_json: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    c_puct: f64,
    seed: c_int,
) -> i64 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_forest_add_root_impl(handle, obs_json, fixed_deck_json, opp_deck_json, iterations, c_puct, seed)
    }));
    match result {
        Ok(Ok(id)) => id as i64,
        // A rejected root is routine (deck-selection steps have no `select`),
        // but the caller only sees -1.  Print the reason so a systematic
        // rejection is diagnosable without rebuilding with logging.
        Ok(Err(e)) => {
            eprintln!("puct_forest_add_root: {e}");
            -1
        }
        Err(_) => {
            eprintln!("puct_forest_add_root: panicked");
            -1
        }
    }
}

fn puct_forest_add_root_impl(
    handle: i64,
    obs_json: *const c_char,
    fixed_deck_json: *const c_char,
    opp_deck_json: *const c_char,
    iterations: c_int,
    c_puct: f64,
    seed: c_int,
) -> Result<usize, String> {
    if handle == 0 {
        return Err("null handle".into());
    }
    let h = unsafe { &mut *(handle as *mut PuctForestHandle) };

    let obs_str = unsafe { cstr_to_str(obs_json) };
    let fixed_str = unsafe { cstr_to_str(fixed_deck_json) };
    let opp_str = unsafe { cstr_to_str(opp_deck_json) };

    let obs: serde_json::Value = serde_json::from_str(&obs_str)
        .map_err(|e| format!("obs_json parse: {e}"))?;
    let fixed_deck: Vec<i32> = serde_json::from_str(&fixed_str)
        .map_err(|e| format!("fixed_deck_json parse: {e}"))?;
    let opp_deck_template: Vec<i32> = serde_json::from_str(&opp_str)
        .map_err(|e| format!("opp_deck_json parse: {e}"))?;

    if obs.get("select").and_then(|s| s.as_object()).is_none() {
        return Err("deck selection step".into());
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let guesses = guessing::build_guesses(&obs_str, &fixed_deck, &opp_deck_template, &mut rng)
        .map_err(|e| format!("build_guesses: {e}"))?;

    // An empty sbi is not a recoverable search — `SearchBegin` dereferences
    // the pointer and takes the whole process down with SIGSEGV, which no
    // `catch_unwind` above can turn back into an error code.  Reject it here.
    let sbi = obs.get("search_begin_input")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if sbi.is_empty() {
        return Err("empty search_begin_input".into());
    }

    // Use engine 0 for root creation (sequential, before rayon kicks in)
    let engine = h.pool.get(0);
    let root = engine.search_begin(
        sbi,
        &guesses.your_deck, &guesses.your_prize,
        &guesses.opponent_deck, &guesses.opponent_prize,
        &guesses.opponent_hand, &guesses.opponent_active,
        false,
    ).map_err(|e| format!("search_begin: {e}"))?;

    let our_player_index = obs.get("current")
        .and_then(|c| c.get("yourIndex"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let root_obs: engine::SearchObservation =
        serde_json::from_str(&root.observation_json)
            .map_err(|e| format!("root obs parse: {e}"))?;

    let n_options = root_obs.select.as_ref()
        .map(|s| s.option.len()).unwrap_or(0);
    let is_terminal = root_obs.current.as_ref()
        .and_then(|c| c.get("result").and_then(|r| r.as_i64()))
        .map(|r| r != -1).unwrap_or(false);
    let player_role = root_obs.current.as_ref()
        .and_then(|c| c.get("yourIndex").and_then(|v| v.as_i64()))
        .map(|v| v as u8).unwrap_or(0);

    let root_node = PuctNode {
        search_id: root.search_id,
        action: None,
        n_options,
        visits: 1.0,
        total_value: 0.0,
        priors: Vec::new(),
        children: Vec::new(),
        obs_json: root.observation_json.clone(),
        player_role,
        is_terminal,
        terminal_value: None,
    };

    let config = PuctConfig {
        iterations: iterations.max(1) as u32,
        c_puct: if c_puct > 0.0 { c_puct } else { 2.0 },
        seed: seed as u64,
        ..PuctConfig::default()
    };

    Ok(h.forest.add_tree(root_node, our_player_index, config))
}

/// Select up to `batch_size` leaves across all active trees.
///
/// Returns a JSON array of leaf requests (caller must free with
/// `puct_free_result`).
#[no_mangle]
pub unsafe extern "C" fn puct_forest_select_batch(
    handle: i64,
    batch_size: c_int,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_forest_select_batch_impl(handle, batch_size)
    }));
    match result {
        Ok(json) => CString::new(json).unwrap_or_else(|_| CString::new("[]").unwrap()).into_raw(),
        Err(_) => CString::new("[]").unwrap().into_raw(),
    }
}

fn puct_forest_select_batch_impl(handle: i64, batch_size: c_int) -> String {
    if handle == 0 {
        return "[]".to_string();
    }
    let h = unsafe { &mut *(handle as *mut PuctForestHandle) };

    // Step 1: select leaves (no engine needed — pure tree traversal)
    let mut leaves = h.forest.select_batch(batch_size.max(1) as usize);

    // Step 2: realise lazy children (needs engine pool)
    // We need to borrow h.pool immutably and h.forest mutably.
    // Since EnginePool owns a Vec<Engine>, we can get a raw pointer to the
    // engines slice and pass it to realise_leaves.
    let engines_ptr: *const engine::Engine = h.pool.get(0) as *const engine::Engine;
    let n_engines = h.pool.len();
    // SAFETY: engines_ptr points to a contiguous array of `n_engines` Engine
    // values stored in h.pool.  The pool is not modified during realise_leaves.
    let engines_slice = unsafe { std::slice::from_raw_parts(engines_ptr, n_engines) };
    h.forest.realise_leaves(&mut leaves, engines_slice);

    serde_json::to_string(&leaves.iter().map(|l| {
        serde_json::json!({
            "tree_id": l.tree_id,
            "obs_json": l.obs_json,
            "player_role": l.player_role,
            "is_terminal": l.is_terminal,
            "n_options": l.n_options,
        })
    }).collect::<Vec<_>>()).unwrap_or_else(|_| "[]".to_string())
}

/// Expand previously selected leaves with NN priors and values.
///
/// `expansions_json`: JSON array of `{tree_id, priors: [...], value}`.
/// Returns the number of successful expansions.
#[no_mangle]
pub unsafe extern "C" fn puct_forest_expand_batch(
    handle: i64,
    expansions_json: *const c_char,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_forest_expand_batch_impl(handle, expansions_json)
    }));
    match result {
        Ok(n) => n as c_int,
        Err(_) => -1,
    }
}

fn puct_forest_expand_batch_impl(
    handle: i64,
    expansions_json: *const c_char,
) -> usize {
    if handle == 0 {
        return 0;
    }
    let h = unsafe { &mut *(handle as *mut PuctForestHandle) };
    let json_str = unsafe { cstr_to_str(expansions_json) };

    let expansions: Vec<ForestExpansion> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return 0,
    };

    h.forest.expand_batch(&expansions)
}

/// Get results for all trees in the forest.
///
/// Returns a JSON array (caller must free with `puct_free_result`).
#[no_mangle]
pub unsafe extern "C" fn puct_forest_results(handle: i64) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        puct_forest_results_impl(handle)
    }));
    match result {
        Ok(json) => CString::new(json).unwrap_or_else(|_| CString::new("[]").unwrap()).into_raw(),
        Err(_) => CString::new("[]").unwrap().into_raw(),
    }
}

fn puct_forest_results_impl(handle: i64) -> String {
    if handle == 0 {
        return "[]".to_string();
    }
    let h = unsafe { &mut *(handle as *mut PuctForestHandle) };
    let engines_ptr: *const engine::Engine = h.pool.get(0) as *const engine::Engine;
    let n_engines = h.pool.len();
    let engines_slice = unsafe { std::slice::from_raw_parts(engines_ptr, n_engines) };
    let results = h.forest.all_results(engines_slice);
    serde_json::to_string(&results.iter().map(|r| {
        serde_json::json!({
            "tree_id": r.tree_id,
            "visit_counts": r.visit_counts,
            "root_value": r.root_value,
            "iterations": r.iterations,
            "nodes_created": r.nodes_created,
        })
    }).collect::<Vec<_>>()).unwrap_or_else(|_| "[]".to_string())
}

/// Free a PUCT forest handle.
#[no_mangle]
pub unsafe extern "C" fn puct_forest_free(handle: i64) {
    if handle != 0 {
        drop(unsafe { Box::from_raw(handle as *mut PuctForestHandle) });
    }
}

// ── VecEnv C-ABI (Rust-backed parallel game env, via C bridge) ──────────

use vec_env::{FinishedGame, VecEnv, VecEnvConfig};

struct VecEnvHandle { env: VecEnv, }

#[no_mangle]
pub unsafe extern "C" fn vec_env_create(
    n_envs: c_int, lib_path: *const c_char,
    deck_self_json: *const c_char, deck_opp_json: *const c_char,
    our_player: c_int, seed: c_int, host_initialized: c_int,
) -> i64 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let lib_str = unsafe { cstr_to_str(lib_path) };
        let deck_self: Vec<i32> = serde_json::from_str(&unsafe { cstr_to_str(deck_self_json) })
            .map_err(|e: serde_json::Error| format!("deck_self: {e}"))?;
        let deck_opp: Vec<i32> = serde_json::from_str(&unsafe { cstr_to_str(deck_opp_json) })
            .map_err(|e: serde_json::Error| format!("deck_opp: {e}"))?;
        let cfg = VecEnvConfig { n_envs: n_envs.max(1) as usize, our_player: our_player as u8,
                                 seed: seed as u64, ..VecEnvConfig::default() };
        let env = VecEnv::new(&lib_str, &deck_self, &deck_opp, cfg)
            .map_err(|e: String| format!("VecEnv: {e}"))?;
        Ok::<VecEnvHandle, String>(VecEnvHandle { env })
    }));
    match result { Ok(Ok(h)) => Box::into_raw(Box::new(h)) as i64, _ => 0 }
}

#[no_mangle]
pub unsafe extern "C" fn vec_env_poll(handle: i64) -> *mut c_char {
    if handle == 0 { return CString::new("[]").unwrap().into_raw(); }
    let h = unsafe { &mut *(handle as *mut VecEnvHandle) };
    let pending = h.env.poll();
    let json = serde_json::to_string(&pending.iter().map(|p| serde_json::json!({
        "battle_idx": p.battle_idx, "obs_json": p.obs_json,
        "sbi": p.sbi, "select_player": p.select_player,
    })).collect::<Vec<_>>()).unwrap_or("[]".into());
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn vec_env_reply(handle: i64, picks_json: *const c_char) -> c_int {
    if handle == 0 { return 0; }
    let h = unsafe { &mut *(handle as *mut VecEnvHandle) };
    let picks: Vec<Vec<i32>> = match serde_json::from_str(&unsafe { cstr_to_str(picks_json) }) {
        Ok(v) => v, Err(_) => return 0,
    };
    h.env.reply(&picks);
    picks.len() as c_int
}

#[no_mangle]
pub unsafe extern "C" fn vec_env_drain(handle: i64) -> *mut c_char {
    if handle == 0 { return CString::new("[]").unwrap().into_raw(); }
    let h = unsafe { &mut *(handle as *mut VecEnvHandle) };
    let finished = h.env.drain();
    let json = serde_json::to_string(&finished.iter().map(|g| serde_json::json!({
        "battle_idx": g.battle_idx, "reward": g.reward,
        "n_decisions": g.n_decisions, "error": g.error,
    })).collect::<Vec<_>>()).unwrap_or("[]".into());
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn vec_env_free(handle: i64) {
    if handle != 0 { drop(unsafe { Box::from_raw(handle as *mut VecEnvHandle) }); }
}
