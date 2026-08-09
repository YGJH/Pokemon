# 管線架構（中文版）

五個階段，每個階段寫出凍結工件供下一階段讀取。階段之間不傳遞記憶體內資料 — 每個邊界都是一個檔案，這也是階段 2 和階段 3 能夠跳過自身的原因。

```
Kaggle API ──1──▶ raw/<day>/<id>.json     235 GB, 52209 個有效對局
                        │
                  2 (mine, 約 15 分鐘)
                        ▼
              data/vocab.json, archetypes.json,
              engine_card_features.npy, *_static_table.npy
                        │
                  3 (build-shards, 約 35 分鐘)
                        ▼
              data/shards/*.npz + data/meta.parquet
                        │
                  4 (train)          ─── ShardDataset → Policy
                        ▼
              checkpoints_a<N>/ckpt-best.pt + data/il_baselines.json
                        │
                  5 (RL: R1 評論家修復 → R2 PPO)
```

## 階段 1 — 下載

`ptcg_mine/download.py`。Phase 0 選取日期（`sampling.py`），Phase 1 以可恢復、多執行緒方式下載對局，從最新日期開始，這樣即便被限速也能確保硬碟上保留最貼近當前 meta 的日資料。Kaggle API 始終以 `api` 參數注入，因此沒有任何測試會觸及真實網路。吞吐量受每連線節流限制，因此此階段瓶頸在於併發數，而非頻寬。

## 階段 2 — 挖掘（`ptcg_mine/mine.py:run`）

載入並驗證每個對局（`statuses == ["DONE","DONE"]`，≥2 步，2 位玩家），然後：

**排行榜（`stats.py`）** — 每隊，統計所有對局中的遊戲數/勝場。排行榜以威爾遜下界而非原始勝率排序專家，僅統計遊戲數 ≥ `g_min=50` 的隊伍。

**原型聚類（`archetype.py:cluster_decks`）** — 每個牌組列表被正規化為排序後的 60-元組，計數，然後按頻率降序貪婪聚類：一個牌組加入其代表牌組與之多重集 Jaccard ≥ 0.90 的群集，否則開闢一個新群集。

微妙之處在於 ID 穩定性。群集 ID 僅為群集開啟的順序（等於牌組頻率排序），所以一次無種子的重新挖掘會把從未改變成員的群集重新編號，導致 `--archetype-self 0` 靜默地變成另一個牌組。因此 `cluster_decks` 預設從前一次的 `archetypes.json` 獲得種子：基線群集保留其 ID 和代表牌組，新加入者從 `max(id)+1` 開始編號，而在新語料庫中無成員的群集以頻率 0 保留，而非釋放其 ID。代價是獲得種子的群集代表牌組會被凍結，不再追蹤 meta。

**𝒟_self / 𝒟_opp（`select_self_opp`）** — 按頻率選取前 `n_self`/`n_opp` 個原型作為專家的自身/對手牌組。這些是僅追加的列表，順序具有負載語義：`shard_writer` 將信念標籤建立為 `{gid: i for i, gid in enumerate(opp_ids)}`，因此類別索引是位置而非 ID。若重新排序 `opp_ids`，頭部維度保持不變，載入無報錯，但每個類別都會被重新指向。

**詞彙表（`vocab.py`）** — 卡片 ID → 連續索引，PAD=0，UNKNOWN=1，真實 ID 從 2 開始。從語料庫中所有卡片建立，刻意與原型過濾器解耦，使即時對局中有極少 OOV 卡片。

**靜態表（`cards.py`）** — 將引擎的 `all_card_data`/`all_attack` 凍結為 `.npy` 檔案：52 基礎 + 29 能力關鍵字 + 2 計數 + 3 攻擊 × 43，每攻擊 43。

## 階段 3 — 建立分片（`ptcg_il/shard_writer.py:build_shards`）

兩次串流遍歷，因為一個解析後的對局約 14.5 MB，語料庫一次載入將超過 100 GB。

**遍歷 A（`_scan_projections`，forked workers）**讀取每個檔案並保留一個約 1 KB 的行列排行榜，並以 `_is_kept_game` 決定保留/丟棄。結果是 `{episode_id: [(player, won), …]}` 映射；對局主體被釋放。

**遍歷 B** 僅重新打開遍歷 A 保留的對局 — 過濾路徑而非解析後丟棄，是節省時間的關鍵。

**保留規則（`_is_kept_game:75`）** — 自身牌組原型 ∈ `self_ids` 且對手牌組原型 ∈ `opp_ids`。

**決策提取（`_active_decisions:110`）** — 差一規則。觀察狀態為 ACTIVE 時記錄；動作標籤來自 `steps[i+1][p]["action"]`。牌組選擇步驟（`obs["select"]` 為 None）被跳過。

**特徵化（`featurizer.py:featurize`，純 NumPy）** — 每個決策點變為固定 token 佈局：

```
L_STATE = 46 tokens:  CLS + 12 寶可夢 + 30 手牌 + 2 摘要 + 1 場地
O_MAX   = 64 選項 tokens
F_GLOBAL=97 (CLS)  F_POKE=26  F_HAND=7  F_SUM=11  F_OPT=8
```

正規化使用固定除數，絕不用 z-score — 全零行是 PAD 哨兵，遮罩會讀取它，因此學習到的中心化會破壞填充訊號。

**寫入內容** — `.npz` 分片，每片 50,000 樣本，按 `sha256(episode_id) % 100` 分割 → 0–9 驗證集，10–19 測試集，20–99 訓練集。按對局分割確保同一對局的所有決策點，以及雙方玩家的軌跡，都在同一個分割中。

**兩件刻意不寫入的東西**：`sample_weight` — 在訓練時才計算。以及所有 `*_card_feat` 張量：每個都是從凍結表中按 ID 純查找的結果，ID 已在特徵化時計算好，而儲存查找結果曾佔據分片解壓後位元組的 93%（每行 130.6 KB vs 僅 ID 的 10.6 KB）。`CARD_FEAT_SOURCES` 是寫入器和資料集共用的 `feat_key → (id_key, table)` 映射。

**meta.parquet** — 每行一個樣本：`sample_uid, shard, row, episode_id, player, archetype_self, archetype_opp, sel_type, sel_ctx, minCount, maxCount, won, skill_w`。這是讓每個牌組的訓練和每行的加權成為可能的索引，無需觸及分片。

## 階段 4 — 訓練讀取路徑（`ptcg_il/train/dataset.py`）

`ShardDataset.__init__` 將 meta 過濾到指定分割，可選再過濾到一個 `archetype_self`，然後：

**mmap 快取（`_prepare_mmap_cache:363`）** — `np.load` 的 `mmap_mode` 對 `.npz` 是靜默忽略的，所以原生讀取會將每個陣列解壓到每 DataLoader worker 的匿名 RAM 中。改為將每個分片解壓一次到 `shards/.mmap-cache/<shard>.npz/<key>.npy`，使其可共享和可驅逐。在父行程中建立，於 workers fork 之前 — 延遲建立會導致 8 個 workers 同時解壓同一個分片，正是要避免的尖峰。根據分片大小/mtime 自失效。

**卡片特徵重新收集（`_rebuild_card_feats`）** — 被丟棄的張量根據儲存的 ID 從父行程中建構一次的密集靜態表按樣本重建。

**樣本權重（`compute_sample_weights:102`）** —

```
w = w_ctx[sel_ctx]^0.5 · w_arch[archetype_self]^0.5 · w_outcome · w_skill
```

正規化到均值 ≈ 1。`w_ctx`/`w_arch` 是逆頻率平衡，`w_outcome = 1.0` 勝 / `0.6` 敗（敗局降權，不丟棄 — 這保留了均勢盤面資料），`w_skill` 是新的 `skill_w` 欄位，`exp(20·(wilson_lb − 0.5))`，取該行隊伍的威爾遜值。

**模型流程** — `TokenEmbedder`（token 類型 + 卡片 ID + 靜態特徵 → D）→ `Encoder`（對 46 個狀態 token 進行自注意力）→ `PointerHead`（選項 token 對狀態 token 進行交叉注意力）→ 每個選項的 logits，加上從 CLS 出來的 `ValueHead`。多重選擇是自回歸的：訓練時 teacher-forced `multiselect_ce`，推論時貪婪 `select_multi`。

## 階段 5 — RL（`ptcg_rl/`）

R1 評論家修復 → R2 PPO + KL 錨定，在訓練好的 IL 專精模型上進行自我對弈。此處是管線的末端。

---

## 跨領域：戳記機制

階段 2 和階段 3 是（原始語料庫、配置參數、實作程式碼）的純函數。`ptcg_mine/stamp.py` 精確指紋識別這些輸入並記錄 `data/.stamp-<stage>.json`。**檢查位於 `mine.run` 和 `cli.cmd_build_shards` 內部**，而非 shell 中，因此直接呼叫任一模組也會觸發檢查。

它絕不能退化為檔案存在性檢查：卡片 ID 是詞彙索引，原型 ID 是群集索引，因此僅僅存在的工件可能對當前程式碼是靜默錯誤的。因此指紋涵蓋決定輸出的原始碼檔案（例如 `featurizer.py` 的編輯會重建分片），以及上游 `vocab.json`/`archetypes.json` 的內容和呼叫者可變動的配置參數。數值參數被正規化，使 `0.90` 和 `0.9` 不構成變更。每個階段僅在其輸出完整後才蓋戳，因此失敗的運行不會留下任何可信任的東西。
