/// Rust bindings for the C bridge (bridge/bridge.c).
///
/// The bridge wraps libcg's struct-by-value returns into out-pointer
/// calls, avoiding the Rust <-> C++ struct-return ABI mismatch.
/// The bridge functions are "Rust" ABI since they're statically linked.

use std::os::raw::c_int;

use crate::ffi::{FnBattleFinish, FnBattleStart, FnGetBattleData, FnSelect,
                  SerialData, StartData};

extern "C" {
    pub fn bridge_game_init(fn_ptr: unsafe extern "C" fn());
    pub fn bridge_battle_start(
        out: *mut StartData,
        fn_ptr: FnBattleStart,
        cards: *const c_int,
    );
    pub fn bridge_get_battle_data(
        out: *mut SerialData,
        fn_ptr: FnGetBattleData,
        ptr: *mut std::ffi::c_void,
    );
    pub fn bridge_select(
        fn_ptr: FnSelect,
        ptr: *mut std::ffi::c_void,
        picks: *const c_int,
        len: c_int,
    ) -> c_int;
    pub fn bridge_battle_finish(
        fn_ptr: FnBattleFinish,
        ptr: *mut std::ffi::c_void,
    );
}
