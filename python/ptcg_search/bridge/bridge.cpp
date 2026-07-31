/// C++ bridge: wraps libcg struct-by-value returns into out-pointer functions.
///
/// GCC/Clang use sret (hidden first arg) for by-value struct returns,
/// even for `extern "C"` functions > 8 bytes.  Rust's `extern "C"` follows
/// the literal ABI spec and reads from registers instead.  The mismatch
/// produces garbage or segfaults.
///
/// These thin wrappers convert every by-value return into an explicit
/// out-pointer first argument — a calling convention every language agrees on.
///
/// Build:
///   g++ -shared -fPIC -O2 -o libcg_bridge.so bridge.cpp
///
/// The resulting .so does NOT link to libcg.so.  Functions are resolved
/// at call time via dlsym(RTLD_NEXT, ...) so the bridge works with any
/// libcg.so loaded in the process.

#include <stdint.h>

// --- struct layouts (must match Export.cpp exactly) -----------------------

struct StartData {
    void *battle_ptr;
    int   error_player;
    int   error_type;
};

struct SerialData {
    const char  *json;
    const uint8_t *data;
    int           count;
    int           select_player;
};

// --- function pointer types (must match Export.cpp signatures) ------------

typedef void        (*GameInitialize_t)();
typedef struct StartData (*BattleStart_t)(int*);
typedef struct SerialData (*GetBattleData_t)(void*);
typedef int         (*Select_t)(void*, int*, int);
typedef void        (*BattleFinish_t)(void*);
typedef void*       (*AgentStart_t)();

// --- lazy symbol resolution -----------------------------------------------

static GameInitialize_t pGameInitialize = nullptr;
static BattleStart_t    pBattleStart = nullptr;
static GetBattleData_t  pGetBattleData = nullptr;
static Select_t         pSelect = nullptr;
static BattleFinish_t   pBattleFinish = nullptr;
static AgentStart_t     pAgentStart = nullptr;
static int resolved = 0;

static void resolve() {
    if (resolved) return;
    // RTLD_DEFAULT searches all loaded libraries — finds libcg.so
    pGameInitialize  = (GameInitialize_t) dlsym(RTLD_DEFAULT, "GameInitialize");
    pBattleStart     = (BattleStart_t)    dlsym(RTLD_DEFAULT, "BattleStart");
    pGetBattleData   = (GetBattleData_t)  dlsym(RTLD_DEFAULT, "GetBattleData");
    pSelect          = (Select_t)         dlsym(RTLD_DEFAULT, "Select");
    pBattleFinish    = (BattleFinish_t)   dlsym(RTLD_DEFAULT, "BattleFinish");
    pAgentStart      = (AgentStart_t)     dlsym(RTLD_DEFAULT, "AgentStart");
    resolved = 1;
}

// --- out-pointer wrappers (safe ABI for any language) ---------------------

extern "C" {

void bridge_game_initialize() {
    resolve();
    pGameInitialize();
}

void bridge_battle_start(void *out, int *cards) {
    resolve();
    *(StartData*)out = pBattleStart(cards);
}

void bridge_get_battle_data(void *out, void *ptr) {
    resolve();
    *(SerialData*)out = pGetBattleData(ptr);
}

int bridge_select(void *ptr, int *picks, int len) {
    resolve();
    return pSelect(ptr, picks, len);
}

void bridge_battle_finish(void *ptr) {
    resolve();
    pBattleFinish(ptr);
}

} // extern "C"
