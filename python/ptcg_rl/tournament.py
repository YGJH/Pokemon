"""Pure UCT MCTS deck ranking — find the best deck from archetypes.json.
No policy network; pure UCB1 + random rollout.  Each deck plays against
a reference deck.  Ranked by win rate.
"""

import argparse, ctypes, json, sys, time
from pathlib import Path


def _load_search_lib():
    base = Path(__file__).resolve().parent.parent / "ptcg_search"
    for target in ["release", "debug"]:
        so = base / f"target/{target}/libptcg_search.so"
        if so.exists():
            lib = ctypes.CDLL(str(so))
            lib.search_plan.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.search_plan.restype = ctypes.c_void_p
            lib.search_plan_free.argtypes = [ctypes.c_void_p]
            return lib
    raise FileNotFoundError("libptcg_search.so not found")


def _pure_mcts(obs_dict, deck_self, deck_opp, iterations, seed, lib, libcg):
    result_ptr = lib.search_plan(
        json.dumps(obs_dict).encode(), libcg.encode(),
        json.dumps(deck_self).encode(), json.dumps(deck_opp).encode(),
        int(iterations), int(seed), 1,
    )
    raw = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode()
    lib.search_plan_free(result_ptr)
    return json.loads(raw).get("indices", [0])


def _find_libcg():
    base = Path(__file__).resolve().parent.parent.parent
    for sub in ["python/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so",
                "pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so"]:
        p = base / sub
        if p.exists(): return str(p)
    raise FileNotFoundError("libcg.so")


def main():
    p = argparse.ArgumentParser(description="Pure UCT MCTS deck ranking")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--reference", type=int, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "archetypes.json") as f:
        arch_data = json.load(f)

    all_archs = arch_data.get("archetypes", [])
    by_id = {int(a["id"]): a for a in all_archs}
    opp_ids = [int(i) for i in arch_data.get("opp_ids", [])]
    all_ids = sorted(by_id.keys())

    ref_id = args.reference or (opp_ids[0] if opp_ids else all_ids[0])
    ref_deck = [int(c) for c in by_id[ref_id].get("representative", [])]
    print(f"Reference deck: archetype {ref_id}, {len(ref_deck)} cards")

    test_ids = all_ids
    if args.top_n > 0:
        ranked = sorted(all_ids, key=lambda aid: by_id[aid].get("count", 0), reverse=True)
        test_ids = ranked[:args.top_n]

    decks = {}
    for aid in test_ids:
        deck = [int(c) for c in by_id[aid].get("representative", [])]
        if len(deck) == 60 and aid != ref_id:
            decks[f"arch_{aid}"] = deck

    print(f"Testing {len(decks)} decks against reference (arch_{ref_id})")
    print(f"Games per deck: {args.games}, iterations: {args.iterations}")

    lib = _load_search_lib()
    libcg = _find_libcg()
    from ptcg_rl.rust_vec_env import RustVecEnv

    results = {}
    total = len(decks)
    done_count = 0
    t0 = time.perf_counter()

    from rich.progress import (
        BarColumn, Progress, TaskProgressColumn,
        TextColumn, TimeElapsedColumn,
    )
    pbar = Progress(
        TextColumn("  [bold]{task.description}[/]"),
        BarColumn(), TaskProgressColumn(),
        TextColumn("• {task.fields[info]}"),
        TimeElapsedColumn(),
    )
    task = pbar.add_task(f"{total} decks × {args.games}g", total=total, info="")

    with pbar:
        for name, deck in sorted(decks.items()):
            done_count += 1
            wins, losses = 0, 0

            for seat in (0, 1):
                d_self = deck if seat == 0 else ref_deck
                d_opp = ref_deck if seat == 0 else deck
                per_seat = max(1, args.games // 2)

                with RustVecEnv(
                    deck_self=d_self, deck_opp=d_opp,
                    n_envs=4, our_player=seat, seed=done_count,
                ) as env:
                    done = 0
                    while done < per_seat:
                        pending = env.poll()
                        if not pending:
                            for t in env.drain():
                                if t.reward > 0: wins += 1
                                elif t.reward < 0: losses += 1
                                done += 1
                            if not pending:
                                time.sleep(0.001)
                            continue

                        picks_list = []
                        for p in pending:
                            obs = json.loads(p["obs_json"])
                            obs["search_begin_input"] = p.get("sbi", "")
                            picks = _pure_mcts(
                                obs,
                                d_self if p["select_player"] == seat else d_opp,
                                d_opp if p["select_player"] == seat else d_self,
                                args.iterations, seat * 1000 + done, lib, libcg,
                            )
                            picks_list.append(picks)

                        env.reply(picks_list)
                        for t in env.drain():
                            if t.reward > 0: wins += 1
                            elif t.reward < 0: losses += 1
                            done += 1

            wr = wins / max(wins + losses, 1)
            results[name] = {"wins": wins, "losses": losses, "wr": wr}
            pbar.update(task, completed=done_count,
                        info=f"last={name} wr={wr:.3f}")

    print("\n=== Deck Ranking (pure UCT vs arch_{}) ===".format(ref_id))
    for rank, (name, r) in enumerate(
        sorted(results.items(), key=lambda x: -x[1]["wr"]), 1
    ):
        print(f"  {rank:>3}. {name:<12}  wr={r['wr']:.3f}  {r['wins']}W/{r['losses']}L")


if __name__ == "__main__":
    main()
