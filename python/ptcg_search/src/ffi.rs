/// Raw FFI bindings to libcg.so — mirrors sim.py ctypes definitions.
///
/// All functions are loaded dynamically via libloading.  The library is
/// leaked so Symbol references remain valid for the process lifetime —
/// but leaked *once per path*, via `LIB_CACHE`, not once per load.

use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// ── Function pointer type aliases ────────────────────────────────────────

pub type FnAgentStart = unsafe fn() -> *mut c_void;
pub type FnGameInitialize = unsafe fn();

// Battle API types (used via bridge/bridge.c — must be extern "C")
pub type FnBattleStart = unsafe extern "C" fn(cards: *const c_int) -> StartData;
pub type FnGetBattleData = unsafe extern "C" fn(battle_ptr: *mut c_void) -> SerialData;
pub type FnSelect = unsafe extern "C" fn(battle_ptr: *mut c_void, select: *const c_int,
                                select_len: c_int) -> c_int;
pub type FnBattleFinish = unsafe extern "C" fn(battle_ptr: *mut c_void);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct StartData {
    pub battle_ptr: *mut c_void,
    pub error_player: c_int,
    pub error_type: c_int,
}
unsafe impl Send for StartData {}
unsafe impl Sync for StartData {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SerialData {
    pub json: *const c_char,
    pub data: *const u8,
    pub count: c_int,
    pub select_player: c_int,
}
unsafe impl Send for SerialData {}
unsafe impl Sync for SerialData {}

// Search API types

pub type FnSearchBegin = unsafe fn(
    agent_ptr: *mut c_void, sbi_json: *const c_char, sbi_len: c_int,
    your_deck: *const c_int, your_prize: *const c_int,
    opp_deck: *const c_int, opp_prize: *const c_int,
    opp_hand: *const c_int, opp_active: *const c_int,
    manual_coin: c_int,
) -> *mut c_char;

pub type FnSearchStep = unsafe fn(
    agent_ptr: *mut c_void, search_id: i64,
    select: *const c_int, select_len: c_int,
) -> *mut c_char;

pub type FnSearchEnd = unsafe fn(agent_ptr: *mut c_void);
pub type FnSearchRelease = unsafe fn(agent_ptr: *mut c_void, search_id: i64);

// ── CgLib ────────────────────────────────────────────────────────────────

pub struct CgLib {
    pub agent_start: Symbol<'static, FnAgentStart>,
    pub game_initialize: Symbol<'static, FnGameInitialize>,
    pub battle_start: Symbol<'static, FnBattleStart>,
    pub get_battle_data: Symbol<'static, FnGetBattleData>,
    pub select: Symbol<'static, FnSelect>,
    pub battle_finish: Symbol<'static, FnBattleFinish>,
    pub search_begin: Symbol<'static, FnSearchBegin>,
    pub search_step: Symbol<'static, FnSearchStep>,
    pub search_end: Symbol<'static, FnSearchEnd>,
    pub search_release: Symbol<'static, FnSearchRelease>,
}

/// Process-wide cache of loaded libraries, keyed by path.
///
/// The `Symbol<'static, _>` fields above require a library that outlives
/// every `CgLib`, which is why one is leaked.  Caching makes that leak
/// happen *once per path* rather than once per `CgLib::load`: the pool
/// loads one `CgLib` per engine, so an uncached leak scaled with the
/// number of engines ever created.  `dlopen` on an already-open path only
/// bumps a refcount, so the mapping was never duplicated — but the
/// `Library` handle and its refcount were, permanently.
static LIB_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, &'static Library>>,
> = std::sync::OnceLock::new();

fn load_library_cached(lib_path: &str) -> Result<&'static Library, Box<dyn std::error::Error>> {
    let cache = LIB_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().map_err(|_| "LIB_CACHE poisoned")?;
    if let Some(lib) = guard.get(lib_path) {
        return Ok(lib);
    }
    let lib: &'static Library = Box::leak(Box::new(unsafe { Library::new(lib_path)? }));
    guard.insert(lib_path.to_string(), lib);
    Ok(lib)
}

impl CgLib {
    pub unsafe fn load(lib_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let lib = load_library_cached(lib_path)?;

        macro_rules! sym {
            ($name:ident, $ty:ty) => {
                std::mem::transmute(lib.get::<$ty>(stringify!($name).as_bytes())?)
            };
        }

        Ok(CgLib {
            game_initialize: sym!(GameInitialize, FnGameInitialize),
            agent_start: sym!(AgentStart, FnAgentStart),
            battle_start: sym!(BattleStart, FnBattleStart),
            get_battle_data: sym!(GetBattleData, FnGetBattleData),
            select: sym!(Select, FnSelect),
            battle_finish: sym!(BattleFinish, FnBattleFinish),
            search_begin: sym!(SearchBegin, FnSearchBegin),
            search_step: sym!(SearchStep, FnSearchStep),
            search_end: sym!(SearchEnd, FnSearchEnd),
            search_release: sym!(SearchRelease, FnSearchRelease),
        })
    }
}

pub unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() { return String::new(); }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}
