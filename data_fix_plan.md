● 兩個 agents 都回傳了詳細分析。以下是我的完整看法。

  ---
  問題 1：ATTACHMENT COLLISION（39.78%）

  根因
  
  當兩張完全相同的卡（例如兩張基本火能量）附在同一個 Pokemon 上時，pointer head 收到的是完全相同的輸入：

  - opt_src_idx 相同 → 指向同一個 Pokemon token row → 相同的 gathered src vector
  - opt_card_id 相同 → 相同的 card_enc(opt_card_feat) vector
  
  兩個選項在模型眼中無法區分，模型只能隨機猜。

  不同卡牌（如火能量 vs 水能量）不受影響——opt_card_id 已經能區分它們，spec A.7 的 card_enc 就是為此設計的。

  我的看法

  這問題的實際影響可能比數字看起來小。原因是：兩張完全相同的能量卡，從遊戲結果來看，選擇哪一張沒有差異——棄掉任何一張基本火能量對後續遊戲狀態的影響是一樣的。模型隨機選一個，不會影響勝率
  。

  但有一個場景會有影響：當其中一張能量是「剛貼上去、這回合不能用」vs「上回合貼的、可以用」時，遊戲邏輯上可能有差別（取決於遊戲引擎的具體實作）。還有 tool
  card（裝備卡）的場景：兩張相同的 Choice Band 在牌組中只會有一張，不太會碰撞。

  我的建議是：先量測再決定。目前訓練的 top-1 accuracy 在 attachment 類選項上的表現如何？如果 attachment 類選項的 accuracy 已經很高（例如 >80%），那碰撞的 39.78%
  中大部分可能是「無差異」的選擇，進一步優化的 ROI 不高。如果 attachment accuracy 明顯低於其他選項類型，才值得投入 A.7 的 token 化改造。

  A.7 的改造幅度不小——需要改 state token layout、featurizer、encoder、shard format，所有 shard 要重建，所有模型要重訓。這是 2-3 天的工作量。

  ---
  問題 2：Reference round-trip QA gate skipped
  
  根因（有兩個）

  第一個是 bug：_build_simple_ref_from_obs() 在 qa.py:363 使用 "cardId" 查找卡牌 ID，但真實 observation 的 state 區塊使用 "id"。model/ref_map.py:112 的文件甚至明確記錄了這個差異：

  ▎ Note the asymmetry with options: state containers key the card id as card["id"], while options (rarely) use cardId.

  所以即使提供了 observation 資料，gate 也會靜默建立空的 ref map，永遠抓不到任何錯誤。測試通過是因為 test_qa.py 裡的人造資料也用了 "cardId"。

  第二個是管線缺失：shard_writer.py 在 featurize 完就丟棄了 obs dict，沒有任何程式碼寫入 data/observations/。

  我的看法

  這個 gate 的設計意圖是對的——featurizer 的 pointer encoding（opt_src_idx 指向哪張卡）是一個很容易出錯的地方，off-by-one 會讓所有訓練樣本的 source pointer 都錯位。但 gate
  從未被真正啟用過，我們也沒遇到過這類 bug（或遇到了但沒發現）。

  修復分兩層：

  1. Bug fix（必做）：把 "cardId" → "id"，修測試。一行改動。
  2. 管線補全（選做）：在 shard_writer.py 中加幾行，把 obs dict 寫成 JSONL。
  1. Attachment collision：先加 per-SelectType accuracy 量測，看到數據再決定是否投入 A.7
  2. Reference round-trip gate：完整修復 — bug fix + 管線補全 + 反向查找

  讓我呈現設計方案。

  ---
  設計方案

  Part A：Per-SelectType accuracy 分解

  改動範圍：python/ptcg_il/train/eval.py

  在現有 evaluate 函數的輸出中，加入按 opt_type 分類的 top-1/top-3 accuracy。選項類型對照：

  ┌──────────┬────────────────────────────────────────────────────────────────────────────┐
  │ opt_type │                                    含義                                    │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 3,4,5,6  │ CARD, TOOL_CARD, ENERGY_CARD, ENERGY（= attachment 類，受 collision 影響） │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 7        │ PLAY（手牌打出）                                                           │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 8        │ ATTACH（貼能量）                                                           │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 9        │ EVOLVE                                                                     │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 10       │ ABILITY                                                                    │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 11       │ DISCARD                                                                    │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 12       │ RETREAT                                                                    │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 13       │ ATTACK                                                                     │
  ├──────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 0,1,2    │ END, YES, NO（常數型）                                                     │
  └──────────┴────────────────────────────────────────────────────────────────────────────┘

  每個 sample 有多個選項（multi-select），accuracy 計算要考慮整個 action 序列的正確率，以及 per-option-position 的正確率。

  輸出示例：
  Per-type top-1 accuracy:
    ATTACHMENT (types 3-6):  2045/3120 = 65.5%
    PLAY (type 7):            890/1200 = 74.2%
    ATTACK (type 13):         567/700  = 81.0%
    ...

  Part B：Reference round-trip gate 修復

  三個子任務：

  B1. Bug fix — qa.py _build_simple_ref_from_obs，line 363-381：把所有 "cardId" 改成 "id"。同時修正 test_qa.py 中的對應測試資料。

  B2. 管線補全 — shard_writer.py：在 _write_shard 或 build_shards 的 decision loop 中，對每個成功 featurize 的 sample，把原始 obs dict 寫入對應的 data/observations/<shard>_obs.jsonl。

  B3. 反向查找 — qa.py check_reference_roundtrip：當 shard 只有 opt_card_feat（舊格式）而沒有 opt_card_id 時，載入 all_card_feat 靜態表，用 opt_card_feat 向量反查
  card_id。比對方式是逐行 L2 distance 找最小值；PAD 列（全零）直接跳過。

  要注意的是 opt_card_feat 是直接從 all_card_feat[card_id] 複製出來的 float32，所以精確匹配是可行的（用 np.allclose 或 bytes equality）。

  ---
