/* Bridge: wrap libcg struct-by-value returns into out-pointer functions.
 *
 * The libcg functions are NOT linked at build time.  Instead we declare
 * them with the correct signatures and call through function pointers
 * that the Rust side resolves via libloading.  This avoids both the
 * struct-return ABI mismatch AND the need to link to libcg.so.
 *
 * Every function takes an explicit out-pointer first argument. */

#include <stdint.h>

struct StartData {
    void *battle_ptr;
    int   error_player;
    int   error_type;
};

struct SerialData {
    const char   *json;
    const uint8_t *data;
    int            count;
    int            select_player;
};

typedef void (*game_init_t)(void);
typedef struct StartData  (*battle_start_t)(int*);
typedef struct SerialData (*get_battle_data_t)(void*);
typedef int  (*select_t)(void*, int*, int);
typedef void (*battle_finish_t)(void*);

/* ── out-pointer wrappers ─────────────────────────────────────────────── */

void bridge_game_init(game_init_t fn) { fn(); }

void bridge_battle_start(void *out, battle_start_t fn, int *cards) {
    *(struct StartData *)out = fn(cards);
}

void bridge_get_battle_data(void *out, get_battle_data_t fn, void *ptr) {
    *(struct SerialData *)out = fn(ptr);
}

int bridge_select(select_t fn, void *ptr, int *picks, int len) {
    return fn(ptr, picks, len);
}

void bridge_battle_finish(battle_finish_t fn, void *ptr) { fn(ptr); }
