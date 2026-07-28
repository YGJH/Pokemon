/// Safe(r) wrapper around libcg.so search API.
///
/// Manages the agent lifecycle (AgentStart → … → SearchEnd) and translates
/// Rust types ↔ C FFI calls.  All operations on one Engine instance share a
/// single `agent_ptr`.

use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use crate::ffi::{cstr_to_string, CgLib};

/// Result of a single search step: the next observation JSON and its search_id.
#[derive(Debug)]
pub struct SearchResult {
    pub search_id: i64,
    /// Raw JSON observation string from the engine (ApiResult JSON).
    pub observation_json: String,
}

/// Safe wrapper around a loaded libcg.so + active agent session.
///
/// # Lifecycle
///
/// ```text
/// Engine::load(path, init_game) → [GameInitialize], AgentStart
///   search_begin(…) → root SearchResult
///     search_step(id, select) → next SearchResult  (repeat)
///     search_release(id)                            (cleanup unused branches)
///   search_end()                                     (called on Drop)
/// ```
pub struct Engine {
    cg: CgLib,
    agent: *mut c_void,
}

impl Engine {
    /// Load libcg.so, call `AgentStart`, and call `GameInitialize` only when
    /// `init_game` is true.
    ///
    /// `GameInitialize` **must be called exactly once per process.** libcg.so
    /// registers into a fixed-capacity global table, and a second call throws the
    /// C++ `std::runtime_error("buffer full. capacity:7")`.  A C++ exception
    /// crossing the FFI boundary is not something Rust can catch ("fatal runtime
    /// error: Rust cannot catch foreign exceptions"), so the *whole process* dies
    /// with SIGABRT — not a recoverable `Err`.
    ///
    /// When we are reached via `search_plan` from a Python host, `cg.sim` already
    /// called `GameInitialize` at import time, and `dlopen` on the same path hands
    /// back that same already-initialized object.  Such a host must pass
    /// `init_game = false`.  A standalone Rust process that owns the library
    /// passes `true`.
    ///
    /// # Safety
    ///
    /// `lib_path` must point to a valid libcg.so whose ABI matches.
    pub unsafe fn load(
        lib_path: &str,
        init_game: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cg = unsafe { CgLib::load(lib_path)? };

        if init_game {
            unsafe { (cg.game_initialize)() };
        }

        let agent = unsafe { (cg.agent_start)() };
        if agent.is_null() {
            return Err("AgentStart returned null".into());
        }

        Ok(Engine { cg, agent })
    }

    /// Begin a determinized search from the current observation.
    ///
    /// `obs` is the raw `obs_dict` JSON as received by `agent()`.
    /// The other parameters are guesses for hidden information.
    ///
    /// Returns the root `SearchResult` whose `observation.select` contains
    /// the legal options at this decision point.
    pub fn search_begin(
        &self,
        search_begin_input: &str,
        your_deck: &[i32],
        your_prize: &[i32],
        opponent_deck: &[i32],
        opponent_prize: &[i32],
        opponent_hand: &[i32],
        opponent_active: &[i32],
        manual_coin: bool,
    ) -> Result<SearchResult, String> {
        let sbi = search_begin_input.as_bytes();
        let sbi_ptr = sbi.as_ptr() as *const c_char;

        let (yd_p, _) = slice_to_ptr_len(your_deck);
        let (yp_p, _) = slice_to_ptr_len(your_prize);
        let (od_p, _) = slice_to_ptr_len(opponent_deck);
        let (op_p, _) = slice_to_ptr_len(opponent_prize);
        let (oh_p, _) = slice_to_ptr_len(opponent_hand);
        let (oa_p, _) = slice_to_ptr_len(opponent_active);

        // The engine validates array sizes internally from the game state in sbi.
        // We pass the pointer and the engine infers the length.
        let raw = unsafe {
            (self.cg.search_begin)(
                self.agent,
                sbi_ptr,
                sbi_len(sbi),
                yd_p,
                yp_p,
                od_p,
                op_p,
                oh_p,
                oa_p,
                manual_coin as c_int,
            )
        };

        let json = unsafe { cstr_to_string(raw) };
        let result: ApiResult =
            serde_json::from_str(&json).map_err(|e| format!("SearchBegin parse error: {e}"))?;

        if result.error != 0 {
            return Err(format!("SearchBegin error code {}", result.error));
        }

        let state = result
            .state
            .ok_or_else(|| "SearchBegin returned null state".to_string())?;

        Ok(SearchResult {
            search_id: state.search_id,
            observation_json: serde_json::to_string(&state.observation)
                .map_err(|e| format!("serialize observation: {e}"))?,
        })
    }

    /// Step forward in the search tree: choose `select` (option indices) and
    /// get the next observation.
    pub fn search_step(
        &self,
        search_id: i64,
        select: &[i32],
    ) -> Result<SearchResult, String> {
        let (sel_p, sel_l) = slice_to_ptr_len(select);
        let raw = unsafe {
            (self.cg.search_step)(self.agent, search_id, sel_p, sel_l as c_int)
        };

        let json = unsafe { cstr_to_string(raw) };
        let result: ApiResult =
            serde_json::from_str(&json).map_err(|e| format!("SearchStep parse error: {e}"))?;

        if result.error != 0 {
            return Err(format!("SearchStep error code {} (search_id={search_id})", result.error));
        }

        let state = result
            .state
            .ok_or_else(|| "SearchStep returned null state".to_string())?;

        Ok(SearchResult {
            search_id: state.search_id,
            observation_json: serde_json::to_string(&state.observation)
                .map_err(|e| format!("serialize observation: {e}"))?,
        })
    }

    /// Release a search state (free its memory).
    pub fn search_release(&self, search_id: i64) {
        unsafe { (self.cg.search_release)(self.agent, search_id) };
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.cg.search_end)(self.agent) };
    }
}

// ── EnginePool: per-thread Engine access for rayon ──────────────────────

/// A pool of `Engine` instances, one per rayon worker thread.
///
/// RL_SPEC §2.1 established that concurrent battles in one process are
/// independent (the C API takes the pointer explicitly).  We document that
/// evidence here and mark `Engine` as `Send` so rayon can use it.
///
/// # Safety
///
/// `Engine` holds a raw `*mut c_void agent_ptr`.  Each `Engine` instance
/// has its own distinct agent (created by `AgentStart`), and the C API
/// functions take the agent pointer explicitly, so concurrent calls on
/// different `Engine` values do not share mutable state.  This is the
/// §2.1 measurement: 8 threads × 3 games, 0 foreign card ids, 0 errors.
unsafe impl Send for Engine {}

pub struct EnginePool {
    engines: Vec<Engine>,
}

impl EnginePool {
    /// Create a pool by loading `n` independent `Engine` instances.
    ///
    /// Each engine gets its own `AgentStart` call.  `init_game` is only
    /// `true` for the first one (GameInitialize once per process).
    pub fn new(
        lib_path: &str,
        n: usize,
        init_game: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engines = Vec::with_capacity(n);
        for i in 0..n {
            let engine = unsafe { Engine::load(lib_path, init_game && i == 0)? };
            engines.push(engine);
        }
        Ok(EnginePool { engines })
    }

    /// Number of engines in the pool.
    pub fn len(&self) -> usize {
        self.engines.len()
    }

    /// Get a reference to one engine by index (for single-threaded use).
    pub fn get(&self, idx: usize) -> &Engine {
        &self.engines[idx]
    }
}

// ── Helper types (match api.py dataclasses) ────────────────────────────────

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct ApiResult {
    #[allow(dead_code)]
    error: i32,
    state: Option<SearchStateRaw>,
}

#[derive(Debug, Deserialize)]
struct SearchStateRaw {
    #[serde(rename = "searchId")]
    search_id: i64,
    observation: serde_json::Value,
}

/// The observation we get back from a SearchState.
/// We only care about a few top-level fields for the MCTS logic.
#[derive(Debug, Deserialize, Serialize)]
pub struct SearchObservation {
    #[serde(default)]
    pub select: Option<SearchSelect>,
    #[serde(default)]
    pub current: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchSelect {
    #[serde(rename = "type")]
    pub sel_type: i32,
    #[serde(default)]
    pub context: i32,
    #[serde(rename = "minCount", default)]
    pub min_count: i32,
    #[serde(rename = "maxCount", default)]
    pub max_count: i32,
    #[serde(default)]
    pub option: Vec<serde_json::Value>,
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn sbi_len(b: &[u8]) -> c_int {
    b.len() as c_int
}

fn slice_to_ptr_len(s: &[i32]) -> (*const c_int, usize) {
    if s.is_empty() {
        (ptr::null(), 0)
    } else {
        (s.as_ptr(), s.len())
    }
}

