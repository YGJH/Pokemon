MCTS (PUCT) 搜尋蒸餾實作計畫 — 含不完全資訊對手建模

Context

RL_SPEC §8 設計 Design C：MCTS 只用於訓練時搜尋蒸餾，不在部署使用。目標是升級 ptcg_search 從純 UCB1+random rollout → PUCT（Policy-guided UCT），並正確處理不完全資訊。

現有 Belief Model 評估

架構：BeliefModule（GRU over game logs）→ 混入 CLS token → BeliefHeads 預測四個目標：

┌────────┬─────────────────────┬───────────────────────────────────────────────────┐
│  Head  │      預測內容       │                       品質                        │
├────────┼─────────────────────┼───────────────────────────────────────────────────┤
│ arch   │ 對手 archetype 分類 │ 最好 — 6-way，搭配 elimination 很快 collapse      │
├────────┼─────────────────────┼───────────────────────────────────────────────────┤
│ deck   │ 對手 60 張牌分佈    │ 可用，但 sampling 可能給出不合法牌組（6+ 張同卡） │
├────────┼─────────────────────┼───────────────────────────────────────────────────┤
│ hidden │ 尚未看到的隱藏卡牌  │ 最重要但最難訓練 — 訓練標籤隨遊戲進度變化         │
├────────┼─────────────────────┼───────────────────────────────────────────────────┤
│ hand   │ 對手當前手牌        │ 最弱 — 標籤有半回合偏移                           │
└────────┴─────────────────────┴───────────────────────────────────────────────────┘

做得好的：
- GRU over logs 是正確的架構選擇
- OpponentDeckOracle fallback 鏈合理（archetype → card distribution → mirror）
- 訓練標籤來自 replay，是精確的 ground truth
- 所有 card head 共用 CardEncoder，即使沒見過的卡也有合理的 embedding

需要補強的地方（具體發現）：
1. 無對手動作預測 — belief model 只說對手「有什麼牌」，不說對手「會做什麼
2. ArchetypePosterior 未整合進 OpponentDeckOracle.predict() — 文件說是 miimination-based posterior 是最可靠的組件卻沒被用
3. Learned belief heads 沒有量測過的 accuracy — eval 程式碼計算了 val/bel 何記錄的數字
4. 無 explicit visible-card tracking — hidden head 必須自己學會減去已見卡
5. Face-down active 只用 crude heuristic — guessing.rs 從牌組隨機選一張，甚至不保證是 Basic Pokémon
6. Hand label 有半回合偏移 — 預設 w_hand=0.25 承認這個 noise
7. deck_from_distribution 產出 bag of cards 而非 coherent decklist — 不保

核心洞察：Policy 本身就是對手模型

這是關鍵架構決策。 在 MCTS 的對手節點，引擎 search_step 自動處理視角切換 — 回傳的 observation JSON 已經用對手視角（yourIndex 正確、對手手牌可見、我方手牌隱藏）。我們只需要 featurize + policy forward，就能得到對手動作機率分佈。

不需要訓練獨立的對手模型。Policy network 本身就編碼了「給定一個遊戲狀態，玩家會怎麼選」的知識。Featurizer 的 tok_owner 編碼使用相對位置（self=1, opp=2），所以同一個 policy
可以在任何玩家視角上運行。

完整的 MCTS 節點處理

在每個 MCTS 節點:
  engine.search_step() → observation JSON (自動是當前玩家視角)
    → featurize(obs) → batch tensors
    → policy.forward() → P(s,a) priors + V(s) leaf value

  我方節點 (max):
    PUCT: argmax_a [ Q(s,a) + c_puct * P(s,a) * √(ΣN) / (1 + N(s,a)) ]

  對手節點 (min):
    PUCT: argmin_a [ Q(s,a) - c_puct * P_opp(s,a) * √(ΣN) / (1 + N(s,a))
    其中 P_opp 來自 policy 在對手視角的 forward pass

Determinization 流程

對手已見卡牌 → ArchetypePosterior.posterior(observed)
  → 對每個 posterior mass > 0 的 archetype:
       OpponentDeckOracle.predict(obs, rng) → 60-card decklist
       guessing.rs: subtract known cards, deal into deck/hand/prizes
       engine.search_begin(sbi, guessed_deck, guessed_hand, ...)
  → K 個 determinized worlds，每個有自己的 PUCT tree
  → 聚合 K 棵樹的 visit counts → π̃(o), 平均 root_value → Ṽ

架構圖

Python (協調層)                           Rust (加速層, cdylib)
========================                  =============================

MctsBatchDistiller                        PuctForest
  ├─ OpponentDeckOracle (belief model)      ├─ Vec<PuctTree>
  ├─ ArchetypePosterior                     ├─ puct_select_batch() → leav
  ├─ K determinizations per state           ├─ puct_expand_batch()
  └─ 驅動 batched MCTS loop                └─ puct_result() → (visits, va

PolicyActor (複用, 擴充)                    EnginePool
  ├─ featurize(obs_json) → tensors          ├─ Vec<Engine>
  ├─ GPU forward → P(s,a), V(s)             ├─ per-thread acquire
  └─ 我方視角 & 對手視角 都用同一個 policy    └─ unsafe impl Send (per §2.1)

Featurizer (複用, 不修改)                   guessing.rs (擴充)
  └─ tok_owner=1/2 編碼支援角色切換          └─ 用 belief posterior 加權

五個 Phase

Phase 3a: Rust PUCT 核心 + 雙角色 Prior — 1.5 週

ptcg_search/src/mcts.rs:
- PuctConfig：c_puct=2.0, iterations=128, use_opponent_model=true
- MctsNode 增加 priors: Vec<f64>（來自 policy 的 P(s,a)）、is_expanded: b=opponent）
- puct_score() — 我方 max、對手 min：
  - 我方: Q + c_puct * P * √(ΣN) / (1+N)
  - 對手: Q - c_puct * P * √(ΣN) / (1+N)  （min node）
- puct_search_select() → LeafRequest { action_path, leaf_obs_json, player_role, is_terminal, n_options }
- puct_search_expand() — 給定 NN priors + value → 建立 child nodes + back
- puct_search_result() → (visit_counts, root_value)
- 完全移除 rollout — leaf value 由 V(s) 提供
- 記憶體管理：每個節點記錄 search_id，釋放離開子樹的 engine states

ptcg_search/src/lib.rs: 新 C-ABI 函數（per-tree）

Python python/ptcg_rl/search.py（新檔案）:
- PuctSearcher — 包裝單樹 C-ABI
- leaf_batch_forward(leaf_requests, policy, vocab) — 批次 featurize + GPU forward，為我方和對手節點產生各自的 prior

驗證: 1000 個真實決策點，PUCT 的 visit-count 最優動作 ≥ UCT random-rollout baseline

---
Phase 3b: 批次多樹 MCTS Forest + Rayon — 1.5 週

ptcg_search/src/mcts.rs — PuctForest:
- 管理多棵 PUCT tree（跨 determinizations × states）
- puct_forest_select_batch(batch_size) → Vec<LeafRequest> (JSON)
- puct_forest_expand_batch(Vec<LeafExpansion>) — 套用 NN priors/values
- puct_forest_results() → 每棵樹的 visit_counts + root_value
- Drop 釋放全部 engine search_id

ptcg_search/src/engine.rs — EnginePool:
- Vec<Engine> per rayon worker
- unsafe impl Send for Engine（§2.1 證實安全）
- with_engine<F>(&self, f: F) per-thread acquisition

批次推論管線:
MCTS loop:
  1. Rust: collect N=512 leaf observations across all trees
     (每片葉子標記 player_role: 0=我方/1=對手)
  2. Python: featurize 所有 leaves → stack batch
  3. GPU forward: policy._encode + pointer + value
     → priors [N, O_MAX], values [N]
  4. Python: mask 非法選項, normalize priors per-leaf
  5. Rust: expand + backprop (我方 max / 對手 min)

加上 rayon: par_iter_mut() 跨樹進行 select + expand，每個 thread 從 Engin

ptcg_search/Cargo.toml: 新增 rayon

主要風險：search_id 記憶體洩漏。對策：Drop guards + 定期 purge + valgrind

驗證: 100 states × 8 determinizations × 64 iterations < 30 秒，記憶體 < 2

---
Phase 3c: Belief Model 驅動的 Determinization — 0.5 週

修復現有 gap：

1. 整合 ArchetypePosterior 進 OpponentDeckOracle.predict()
  - 目前文件說是 middle rung，但程式碼沒呼叫
  - 加入：posterior = ArchetypePosterior(archetypes); probs = posterior.posterior(observed_cards)
  - Fallback 鏈更新為：learned arch head (≥0.5) → Bayesian posterior (≥0.ribution → mirror
2. 紀錄 learned belief heads 的 accuracy baseline
  - 在 offline eval 中輸出 val/belief_arch_top1, val/belief_deck_mass, va
  - 建立最低門檻，低於門檻的 head 自動 fallback
3. Face-down active Pokémon 改用 belief model 預測
  - guessing.rs 目前隨機選一張（不保證是 Basic Pokémon）
  - 改為從 Python 端預測（可用 hidden head 的 top-K Basic Pokémon）

Python side（search.py）:
- 利用 OpponentDeckOracle.predict() + ArchetypePosterior 產生加權的對手牌
- deck_from_distribution(probs, rng=rng) — 使用 multinomial sampling 產生多樣化的 determinizations
- K 次取樣 → K 個 search_begin roots

ptcg_search/src/guessing.rs 擴充:
- build_guesses_weighted() — 接受 JSON 陣列 [{"deck": [...], "weight": 0.
- 對 deck_from_distribution 輸出做 4-copy rule 檢查（basic Energy 例外）

Rollout 記錄擴充（vec_env.py）:
- Decision 選擇性儲存對手已見卡牌列表（供 belief posterior 計算）

驗證: Weighted determinization 的 root value correlation 比 uniform mirror guess 提升 ≥5%

---
Phase 3d: 離線蒸餾整合進 PPO — 0.5 週

python/ptcg_rl/train.py: --mcts-distill flag
python/ptcg_rl/config.py: MCTS 超參數（mcts_iterations=128, c_puct=2.0, K=8, ρ=0.05, c_π=0.3, c_ṽ=0.3）
python/ptcg_rl/ppo.py: L_search = -c_π * Σ π̃(o) log π_θ(o|s) + c_ṽ * (V_θ(s) - Ṽ)²

驗證: Ablation — MCTS distillation on vs off，對 frozen IL 勝率差異 ≥ 3pt

---
Phase 3e: Multi-Select AR 正確性 — 0.5 週

引擎的 search_step 自動處理 AR 序列 — 每次選一個 option，引擎前進一個 AR step。MCTS 樹自然建模這個過程。

驗證: 1000 個 real multi-select 決策點，零非法序列，visit counts 總和 ≈ i

---
檔案變更摘要

┌────────────────────────────────┬───────┬───────────────────────────────
│              檔案              │ Phase │                        變更                        │
├────────────────────────────────┼───────┼───────────────────────────────
│ ptcg_search/src/mcts.rs        │ 3a-3b │ PUCT node/tree/forest, player_role, remove rollout │
├────────────────────────────────┼───────┼────────────────────────────────────────────────────┤
│ ptcg_search/src/engine.rs      │ 3b    │ EnginePool + unsafe impl Send                      │
├────────────────────────────────┼───────┼───────────────────────────────
│ ptcg_search/src/guessing.rs    │ 3c    │ Weighted determinization                           │
├────────────────────────────────┼───────┼───────────────────────────────
│ ptcg_search/src/lib.rs         │ 3a-3b │ 新 C-ABI functions
├────────────────────────────────┼───────┼───────────────────────────────
│ ptcg_search/Cargo.toml         │ 3b    │ 新增 rayon
├────────────────────────────────┼───────┼───────────────────────────────
│ python/ptcg_rl/search.py       │ 3a-3d │ 新檔案: PuctSearcher, MctsBatchDistiller           │
├────────────────────────────────┼───────┼────────────────────────────────────────────────────┤
│ python/ptcg_rl/config.py       │ 3d    │ MCTS 超參數                                        │
├────────────────────────────────┼───────┼───────────────────────────────
│ python/ptcg_rl/train.py        │ 3d    │ run_r3, L_search 整合                              │
├────────────────────────────────┼───────┼───────────────────────────────
│ python/ptcg_rl/ppo.py          │ 3d    │ search_loss()                                      │
├────────────────────────────────┼───────┼────────────────────────────────────────────────────┤
│ python/ptcg_rl/vec_env.py      │ 3c    │ 記錄對手可見卡牌                                   │
├────────────────────────────────┼───────┼───────────────────────────────
│ python/tests/test_rl_search.py │ 3a-3e │ 新檔案                                             │
├────────────────────────────────┼───────┼───────────────────────────────
│ ptcg_search/tests/             │ 3a-3c │ 新目錄: Rust integration tests                     │
└────────────────────────────────┴───────┴────────────────────────────────────────────────────┘

總預估：約 4.5 週

驗證計畫

1. cargo build --release && cargo test in ptcg_search/
2. uv run pytest python/tests/test_rl_search.py -xvs
3. uv run pytest tests/ && uv run pytest python/tests/（regression check
4. End-to-end: uv run python -m ptcg_rl.train --deck-archetype 0 --il-ckpt checkpoints_a0/ckpt-best.pt --mcts-distill --total-steps 1000