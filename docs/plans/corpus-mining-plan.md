# Plan — Corpus Mining (Phases 0–2)

Implements `TRANSFORMER_IL_SPEC.md` **Appendix D, Phases 0–2 only** (sampling plan + download +
stats/freeze). **Out of scope:** Phase 3/4 featurizer, shards, `meta.parquet`, QA round-trip decode.
Package name: `ptcg_mine/` at repo root. Python via **uv** (`uv add`, `uv run`). Tests with `pytest`.

## Global Constraints (binding — reviewers use these verbatim)

- **Episode format** (verified): `ep["info"]["TeamNames"] = [t0,t1]`; `ep["rewards"] = [r0,r1]` with
  `+1` win / `-1` loss / `0` draw; `ep["steps"]` is `list[[recordP0, recordP1]]`.
- **Deck extraction:** `deck_of(ep,p) = ep["steps"][1][p]["action"]`, which is a 60-int list. Assert
  length 60. (This is the action paired by the off-by-one rule with the `select==None` deck prompt.)
- **Validation:** an episode is usable iff `statuses == ["DONE","DONE"]`, `len(steps) >= 2`,
  `len(rewards) == 2`, and both decks have length 60. Malformed → dropped and counted.
- **Experts:** `EXPERTS = top-K teams by win_rate = wins/games with games >= G_MIN`. A **draw (r==0)
  counts toward `games` but not `wins`**. Defaults `K_EXPERTS=10`, `G_MIN=50` (config).
- **Archetype:** `canon(deck) = tuple(sorted(deck))`; `multiset_counts` = per-id counts;
  `jaccard_multiset(a,b) = Σ min(a_c,b_c) / Σ max(a_c,b_c)` over the union of ids. Cluster **greedily
  in descending exact-deck frequency**: assign a deck to an existing cluster if
  `jaccard_multiset(deck, centroid) >= JACCARD_THRESH` (default 0.90), else start a new cluster;
  the cluster **representative** is its single most-frequent exact decklist.
- **𝒟_self / 𝒟_opp:** `𝒟_self` = top `N_SELF` (default 6, range 3–8) archetypes by frequency **as an
  expert's own deck across all expert games (won and lost)**. `𝒟_opp` = top `N_OPP` (default 6)
  archetypes by frequency **as the opponent's deck in expert games** (may overlap 𝒟_self; no
  mirror-only constraint). `FIXED_DECK` = representative decklist of the 𝒟_self archetype with the
  **highest expert win-rate**.
- **Vocab (decoupled from the archetype filter, §1.2):** `mode="all_corpus"` (locked) = **every
  distinct card id appearing in any sampled episode's decks (both players)**, ranked by corpus
  frequency; remap to contiguous indices with `PAD=0`, `UNKNOWN=1`, real ids `2..V-1`. (Keep an
  `n_vocab` truncation option, default `None` = keep all.)
- **`W_LOST = 0.5`** (config): recorded into artifacts for downstream training weighting. **Not a
  filter** — both won and lost expert games are kept in the corpus.
- **Static tables:** `card_static_row -> float32[52]` and `attack_static_row -> float32[14]` must match
  `TRANSFORMER_IL_SPEC.md` **Appendix A.3** layout exactly, with **Appendix A.2** normalizers
  (`HP_N=400, RETREAT_N=4, ATKDMG_N=350, ATKCOST_N=5`, energy histogram over 12 `EnergyType`).
- **Engine access:** import `all_card_data()` / `all_attack()` from the bundled `cg` package at
  `pokemon-tcg-ai-battle/sample_submission/sample_submission/cg` (Linux x86_64 `libcg.so` present).
- **Determinism & resumability:** sampling is a pure function of `(ids, seed)`; downloads skip files
  already present. **No network in tests** — mock the Kaggle API.

## File structure
```
ptcg_mine/__init__.py
ptcg_mine/config.py       # MineConfig dataclass (all defaults above)
ptcg_mine/episode.py      # load/validate/teams/rewards/deck_of
ptcg_mine/sampling.py     # select_days, deterministic_pick
ptcg_mine/download.py     # list_episode_files, download_episode, download_corpus
ptcg_mine/stats.py        # team_leaderboard, select_experts
ptcg_mine/archetype.py    # canon, jaccard_multiset, cluster_decks, select_self_opp, pick_fixed_deck
ptcg_mine/vocab.py        # build_vocab
ptcg_mine/cards.py        # load_engine, card_static_row, attack_static_row, build_static_tables
ptcg_mine/artifacts.py    # write_vocab_json, write_archetypes_json, write_mining_report
ptcg_mine/mine.py         # CLI orchestrator (Phases 0–2)
tests/…                   # pytest per module
```

---

## Task 1: Scaffolding, config, episode parsing

**Objective.** Create the `ptcg_mine` package, the `MineConfig` dataclass, and the episode parser
(the shared foundation for Phase 2). **Do not** add a per-decision/off-by-one iterator — that is
Phase 3 (out of scope); YAGNI.

**Files:** `ptcg_mine/__init__.py`, `ptcg_mine/config.py`, `ptcg_mine/episode.py`, `tests/test_episode.py`.

**`config.py` — `MineConfig` dataclass** with fields + defaults:
`days: list[str] | None = None` (None ⇒ auto-select recent), `n_days: int = 20`,
`target_episodes: int = 10000`, `seed: int = 0`, `k_experts: int = 10`, `g_min: int = 50`,
`jaccard_thresh: float = 0.90`, `n_self: int = 6`, `n_opp: int = 6`, `w_lost: float = 0.5`,
`vocab_mode: str = "all_corpus"`, `n_vocab: int | None = None`,
`h_max: int = 30`, `o_max: int = 64`, `d_max: int = 60`,
`raw_dir: Path = Path("raw")`, `out_dir: Path = Path("data")`,
`manifest_csv: Path = Path("archive/manifest.csv")`,
`dataset_prefix: str = "kaggle/pokemon-tcg-ai-battle-episodes-"`.

**`episode.py` functions:**
- `load_episode(path) -> dict` — `json.load`.
- `validate_episode(ep) -> bool` — the Validation constraint above.
- `teams(ep) -> tuple[str,str]`; `rewards(ep) -> tuple[int,int]`.
- `deck_of(ep, p: int) -> list[int]` — `ep["steps"][1][p]["action"]`; raise `ValueError` if not len 60.

**Tests** (`tests/test_episode.py`) use the real fixture
`archive/sample_episodes/80169582.json`: assert `teams == ("shikisoukan","cocoaAI")`,
`rewards == (-1, 1)`, `validate_episode is True`, `len(deck_of(ep,0)) == 60` and `== 60` for p=1.
Add a synthetic malformed episode (e.g. `statuses=["ERROR","DONE"]` or a 59-card deck) → `validate`
False / `deck_of` raises.

**Done when:** package imports, `pytest tests/test_episode.py` green, output pristine.

---

## Task 2: Phase 0 sampling + Phase 1 download

**Objective.** Deterministic episode sampling and a resumable, mocked-in-tests downloader.

**Files:** `ptcg_mine/sampling.py`, `ptcg_mine/download.py`, `tests/test_sampling.py`,
`tests/test_download.py`.

**`sampling.py`:**
- `select_days(manifest_df, n_days) -> list[str]` — the `n_days` most recent rows of
  `archive/manifest.csv` (by `date`), returned as day strings `YYYY-MM-DD`. (Recency first; the
  spec's "high top_avg_score" bias is satisfied by recency here — keep it simple.)
- `deterministic_pick(ids: list[str], k: int, seed: int) -> list[str]` — stable hash each id with the
  seed (e.g. `hashlib.sha1(f"{seed}:{id}".encode())`), sort by hash, take first `min(k, len)`.
  **Pure and reproducible**: same `(ids,k,seed)` ⇒ same output regardless of input order.

**`download.py`** (Kaggle API injected for testability — accept an `api` object):
- `list_episode_files(api, slug) -> list[str]` — page through `api.dataset_list_files(slug,
  page_token=...)`, collecting `<id>.json` names until no `nextPageToken`.
- `download_episode(api, slug, filename, dest_dir) -> Path` — skip if `dest_dir/filename` exists
  (resume); else call the api download and place the file at `dest_dir/filename`.
- `download_corpus(config, api) -> "pandas.DataFrame"` — for each `select_days` day: build slug from
  `config.dataset_prefix + day`, `list_episode_files`, `deterministic_pick` per-day quota
  (`target_episodes // n_days`), download into `config.raw_dir/<day>/`, with a bounded thread pool and
  simple retry/backoff on exceptions. Return a manifest DataFrame `[day, episode_id, path, bytes, ok]`
  and also write it to `config.out_dir/downloaded.parquet`.

**Tests** mock the api (a fake object whose `dataset_list_files` returns paged fake ids and whose
download writes a small dummy file): `deterministic_pick` reproducible + order-independent + correct
count; `list_episode_files` concatenates pages; `download_episode` skips existing files;
`download_corpus` produces the manifest and creates `raw/<day>/…`. **No real network.**

**Done when:** `pytest tests/test_sampling.py tests/test_download.py` green, pristine.

---

## Task 3: Phase 2 stats & archetype selection

**Objective.** Team leaderboard → experts; deck archetype clustering → 𝒟_self/𝒟_opp/FIXED_DECK.
Operate over an in-memory list of **parsed episodes** (use `episode.py`); callers load the raw cache.

**Files:** `ptcg_mine/stats.py`, `ptcg_mine/archetype.py`, `tests/test_stats.py`,
`tests/test_archetype.py`.

**`stats.py`:**
- `team_leaderboard(episodes) -> dict[str, dict]` (or DataFrame) with per-team `games`, `wins`,
  `win_rate`; draw counts in `games` only.
- `select_experts(leaderboard, k_experts, g_min) -> list[str]` — top-K by win_rate among teams with
  `games >= g_min`, deterministic tie-break by name.

**`archetype.py`:**
- `canon(deck) -> tuple[int,...]` = `tuple(sorted(deck))`.
- `multiset_counts(deck) -> collections.Counter`.
- `jaccard_multiset(a, b) -> float` per the constraint (handle empty → 0.0).
- `cluster_decks(deck_freq: dict[tuple,int], thresh) -> list[Archetype]` — greedy desc-frequency; an
  `Archetype` carries an id, the representative canon deck, member canons, and total frequency.
- `assign_archetype(deck, archetypes) -> int | None` — best cluster by jaccard≥thresh, else None.
- `select_self_opp(episodes, experts, archetypes, n_self, n_opp) -> (list[int], list[int])` returning
  archetype-id lists for 𝒟_self and 𝒟_opp per the constraint (won+lost expert games).
- `pick_fixed_deck(episodes, experts, self_ids, archetypes) -> list[int]` — representative decklist
  (60 ids) of the 𝒟_self archetype with the highest expert win-rate.

**Tests** use small **synthetic** episode dicts (minimal shape: `info.TeamNames`, `rewards`,
`steps[1][p].action` decks) for deterministic asserts: leaderboard counts incl. a draw;
`jaccard_multiset` on known counters (identical→1.0, disjoint→0.0, 57/60 shared→≥0.9); clustering
merges decks differing by ≤3 cards and separates decks differing by ~10; `select_experts` threshold;
`pick_fixed_deck` returns a 60-int list.

**Done when:** `pytest tests/test_stats.py tests/test_archetype.py` green, pristine.

---

## Task 4: Phase 2 vocab, static tables, artifact writers, CLI

**Objective.** Build the corpus vocab and the engine-derived static feature tables, write the frozen
artifacts, and wire Phases 0–2 into a CLI. Depends on Tasks 1–3.

**Files:** `ptcg_mine/vocab.py`, `ptcg_mine/cards.py`, `ptcg_mine/artifacts.py`, `ptcg_mine/mine.py`,
`tests/test_vocab.py`, `tests/test_cards.py`, `tests/test_artifacts.py`.

**`vocab.py`:**
- `build_vocab(episodes, mode="all_corpus", n_vocab=None) -> dict` returning `{"id_to_index": {card_id:
  idx}, "index_to_id": [...], "freq": {card_id: count}, "size": V}` with `PAD=0`, `UNKNOWN=1`, real
  ids from index 2 ordered by descending corpus frequency (tie-break by id). `all_corpus` keeps every
  distinct id; `n_vocab` (if set) truncates to the top-N. Include the frequency-coverage list.

**`cards.py`:**
- `load_engine() -> (list[CardData], list[Attack])` — add the bundled `cg` path to `sys.path`, import
  `all_card_data`, `all_attack`, return them.
- `card_static_row(card) -> np.ndarray[52] float32` — **exact Appendix A.3 layout / A.2 normalizers**:
  `[0]=hp/400, [1]=retreatCost/4, [2:9]=cardType one-hot(7), [9:12]=[basic,stage1,stage2],
  [12:24]=energyType one-hot(12), [24:36]=weakness one-hot(12, all-zero=none),
  [36:48]=resistance one-hot(12), [48:52]=[ex,megaEx,tera,aceSpec]`.
- `attack_static_row(attack) -> np.ndarray[14] float32` — `[0]=damage/350, [1:13]=energy-cost
  histogram over 12 EnergyType, [13]=len(energies)/5`.
- `build_static_tables(vocab, card_data, attack_data) -> (card_table[V,52] float32, attack_id_to_index,
  attack_table[A,14] float32)` — row 0 = PAD zeros, row 1 = UNKNOWN = **mean of in-vocab card rows**;
  `A` = distinct attackIds referenced by vocab cards (+ PAD row 0).

**`artifacts.py`:**
- `write_vocab_json(path, vocab, attack_id_to_index, config)` — remaps + norm constants + caps
  (`h_max/o_max/d_max`) + `F_*` dims (`F_CARD=52, F_ATK=14, F_POKE=26, F_HAND=2, F_SUM=11,
  F_GLOBAL=93, F_OPT=6`) + `w_lost`.
- `write_archetypes_json(path, self_ids, opp_ids, archetypes, fixed_deck)` — archetype signatures
  (representative decklists), 𝒟_self / 𝒟_opp id lists, and `FIXED_DECK` (60 ids).
- `write_mining_report(path, leaderboard, experts, archetypes, self_ids, opp_ids, vocab, counts)` — a
  Markdown report: team leaderboard (top teams, win-rates), archetype table, vocab size + coverage,
  kept-game counts. Also save `card_static_table`/`attack_static_table` as `.npy` next to `vocab.json`.

**`mine.py`** — argparse CLI: build `MineConfig` (with overrides + `--skip-download`), optionally
`download_corpus`, load parsed episodes from `raw/`, run leaderboard→experts→cluster→
select_self_opp→fixed_deck→build_vocab→load_engine→build_static_tables, then write the three artifacts
+ `.npy` tables into `out_dir`. Print a concise summary.

**Tests:** `card_static_row` returns shape (52,) with correct one-hot slices for a **known** card
(e.g. engine card id 1 = Basic {G} Energy → cardType=BASIC_ENERGY one-hot, energyType=Grass one-hot,
hp slot 0); `attack_static_row` shape (14,); `build_vocab` remap correctness incl. PAD/UNKNOWN and
descending-frequency order on a synthetic corpus; `build_static_tables` shapes `[V,52]`/`[A,14]`,
row0 zeros; `write_vocab_json` round-trips (write then json-load, keys present). Engine-dependent
tests may call `load_engine()` (engine is available on this machine).

**Done when:** `pytest tests/test_vocab.py tests/test_cards.py tests/test_artifacts.py` green;
`uv run python -m ptcg_mine.mine --help` works; output pristine.
