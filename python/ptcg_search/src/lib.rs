//! ptcg_search — MCTS search planner for Pokémon TCG AI Battle.
//!
//! This crate links against `libcg.so` (the game engine) at runtime and
//! exposes a single C-ABI function `search_plan` for Python to call via
//! `ctypes`.

mod engine;
mod ffi;
mod guessing;
mod mcts;

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

    // Get search_begin_input from observation
    let sbi = obs
        .get("search_begin_input")
        .and_then(|v| v.as_str())
        .unwrap_or("");

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
