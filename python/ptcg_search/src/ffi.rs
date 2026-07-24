/// Raw FFI bindings to libcg.so — mirrors sim.py ctypes definitions exactly.
///
/// All functions are loaded dynamically via libloading at runtime.
/// The library is leaked (Box::leak) so Symbol references remain valid for
/// the lifetime of the process.  This is sound because the library is
/// loaded once and never unloaded.

use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// ── C function pointer type aliases (match sim.py signatures) ──────────────
pub type FnAgentStart = unsafe fn() -> *mut c_void;
pub type FnGameInitialize = unsafe fn();
#[allow(dead_code)]
pub type FnBattleFinish = unsafe fn(battle_ptr: *mut c_void);

// SearchBegin: (agent_ptr, sbi_json, sbi_len, your_deck*, your_prize*,
//                opp_deck*, opp_prize*, opp_hand*, opp_active*, manual_coin) -> char*
pub type FnSearchBegin = unsafe fn(
    agent_ptr: *mut c_void,
    sbi_json: *const c_char,
    sbi_len: c_int,
    your_deck: *const c_int,
    your_prize: *const c_int,
    opp_deck: *const c_int,
    opp_prize: *const c_int,
    opp_hand: *const c_int,
    opp_active: *const c_int,
    manual_coin: c_int,
) -> *mut c_char;

// SearchStep: (agent_ptr, search_id, select*, select_len) -> char*
pub type FnSearchStep = unsafe fn(
    agent_ptr: *mut c_void,
    search_id: i64,
    select: *const c_int,
    select_len: c_int,
) -> *mut c_char;

pub type FnSearchEnd = unsafe fn(agent_ptr: *mut c_void);
pub type FnSearchRelease = unsafe fn(agent_ptr: *mut c_void, search_id: i64);

/// Holds dynamically-loaded libcg.so function pointers.
///
/// The underlying `Library` is leaked so all `Symbol` references live for `'static`.
pub struct CgLib {
    pub agent_start: Symbol<'static, FnAgentStart>,
    pub game_initialize: Symbol<'static, FnGameInitialize>,
    pub search_begin: Symbol<'static, FnSearchBegin>,
    pub search_step: Symbol<'static, FnSearchStep>,
    pub search_end: Symbol<'static, FnSearchEnd>,
    pub search_release: Symbol<'static, FnSearchRelease>,
}

impl CgLib {
    /// Load libcg.so from `lib_path`, resolve all required symbols.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib_path` points to a valid libcg.so build
    /// whose ABI matches the declared signatures.  The library is leaked
    /// and never unloaded.
    pub unsafe fn load(lib_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Leak the library so symbols live for 'static.
        let lib = Box::leak(Box::new(Library::new(lib_path)?));

        // Safety: we've leaked `lib`, so all symbols borrowed from it are
        // valid for the remainder of the process lifetime.  We extend the
        // borrow lifetime to 'static via transmute.
        let game_initialize: Symbol<'static, FnGameInitialize> =
            std::mem::transmute(lib.get::<FnGameInitialize>(b"GameInitialize")?);
        let agent_start: Symbol<'static, FnAgentStart> =
            std::mem::transmute(lib.get::<FnAgentStart>(b"AgentStart")?);
        let search_begin: Symbol<'static, FnSearchBegin> =
            std::mem::transmute(lib.get::<FnSearchBegin>(b"SearchBegin")?);
        let search_step: Symbol<'static, FnSearchStep> =
            std::mem::transmute(lib.get::<FnSearchStep>(b"SearchStep")?);
        let search_end: Symbol<'static, FnSearchEnd> =
            std::mem::transmute(lib.get::<FnSearchEnd>(b"SearchEnd")?);
        let search_release: Symbol<'static, FnSearchRelease> =
            std::mem::transmute(lib.get::<FnSearchRelease>(b"SearchRelease")?);

        Ok(CgLib {
            agent_start,
            game_initialize,
            search_begin,
            search_step,
            search_end,
            search_release,
        })
    }
}

/// Helper: convert a `*mut c_char` returned by the engine into a Rust String.
pub unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}
