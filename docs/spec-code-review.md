# 代码与规格文档差异报告

基于 `AGENT_SPEC.md`（环境 I/O 参考）和 `TRANSFORMER_IL_SPEC.md`（权威 IL 规格，附录 A–D）对 `python/` 目录下所有代码进行的系统性审查。

---

## 严重 — 会导致运行时错误

### 1. `orjson` 被导入但未声明为依赖项

**文件**: `python/ptcg_mine/artifacts.py:7`

```python
import orjson
```

`pyproject.toml` 中列出的依赖为 `kaggle`、`kagglehub`、`pandas`、`pyarrow`，没有 `orjson`。它可能作为传递依赖存在，但在干净的安装环境中导入会失败。

**修复**: 将 `orjson` 添加到 `pyproject.toml` 的 dependencies，或切换到标准库的 `json` 模块（数据量小，`orjson` 在这里没有实质性的性能提升）。

---

### 2. `FIXED_DECK` 键名大小写不匹配 — 写入用小写，读取用大写

**写入** (`python/ptcg_mine/artifacts.py:62`):
```python
"fixed_deck": fixed_deck,       # 小写
```

**读取** (`python/ptcg_il/qa.py:784`):
```python
return data.get("FIXED_DECK")   # 大写 — 永远返回 None
```

**读取** (`python/ptcg_il/cli.py:233`):
```python
artifacts["fixed_deck"] = arch.get("FIXED_DECK", list(range(60)))
# 找不到 "FIXED_DECK"，回退到 list(range(60))
```

后果：`qa.py` 的 `load_fixed_deck()` 永远返回 `None`，导致牌组合法性 QA 检查被静默跳过。`cli.py` 回退到一个无意义的 `list(range(60))`。

**修复**: 统一大小写。规格使用 `FIXED_DECK`（大写）。将 `artifacts.py:62` 改为 `"FIXED_DECK": fixed_deck`。

---

## 高 — 行为错误或功能缺失

### 3. `search_planner_agent` 是桩代码 — 永远返回 `[0]`

**文件**: `python/ptcg_il/live_eval.py:191-219`

规格 §5 要求将内置的 `search_begin/step` 规划器作为"真正的、有意义的基线"进行对战评估。但实现只是一个返回 `[0]`（第一个合法选项）的占位符：

```python
def search_planner_agent(obs_dict: dict) -> list[int]:
    ...
    return [0]  # 第一个选项作为回退
```

注释说这需要 `Battle` 对象，但实际的搜索 API（`cg.sim.SearchBegin`/`SearchStep`）使用的是 `SearchState`，并非 `Battle`。引擎搜索可以在运行中的游戏之外调用。

**修复**: 通过 `cg.sim.SearchBegin`/`SearchStep` 实现真正的搜索规划器集成，填入猜测的隐藏状态。

---

### 4. Vocab 从所有对局构建，而非仅从专家对局

**文件**: `python/ptcg_mine/mine.py:139`

规格 D.3.6：
> 按采样语料库中**每个专家对局**的出现频率对所有卡牌 id 进行排序

`mine.py` 调用 `build_vocab(episodes, ...)` 时，`episodes` 是所有已加载的有效对局 — 在专家被计算出来之前。非专家对局的卡牌会稀释频率排序，当 `N_VOCAB` 较小（目标约 300–500）时，这会直接影响哪些卡牌进入词汇表。低水平玩家使用的卡牌可能会取代专家相关的卡牌。

**修复**: 在调用 `build_vocab` 之前，先将 episodes 过滤为至少有一方是专家队伍的对局。

---

### 5. `deck.csv` 从未被挖掘管线生成

规格 C.7 要求提交包包含 `deck.csv(=FIXED_DECK)`。`build_submission_bundle`（`checkpoint.py:191`）会查找 `data_dir / "deck.csv"`。但挖掘管线（`mine.py` → `artifacts.py`）只在 `archetypes.json` 内写入了 `FIXED_DECK` — 从未生成独立的 `deck.csv` 文件。

**修复**: 在 `write_archetypes_json` 时同时写入 `deck.csv`（每行一个整数，或逗号分隔的 60 个整数）。

---

## 中 — 规格/代码不一致，有一定功能影响

### 6. ATTACK 选项的 `opt_tgt_idx` 为狙击攻击设置了值，规格表说应为 `-1`

**文件**: `python/ptcg_il/featurizer.py:527-530`

规格 A.7 表格显示 `ATTACK(13)` → `tgt_idx = -1`。代码在实际存在目标板凳位时设置了 `opt_tgt_idx`（例如"对对手 1 只后备宝可梦也造成 30 点伤害"的狙击效果）：

```python
if in_play_area is not None and in_play_idx is not None:
    opt_tgt_idx[j] = _ref(in_play_area, 1 - your_index, in_play_idx)
```

这是对规格的合理改进 — 目标板凳位是有用的信号。但规格表格应更新以反映此行为。

---

### 7. 奖励卡池化已加入摘要令牌，但规格 B.2 伪代码未体现

**文件**: `python/ptcg_il/model/embed.py:111-114`

规格 §2.1 说"已揭示的奖励卡以相同方式携带"。规格 B.2 伪代码仅展示了弃牌堆池化。代码正确地添加了奖励卡池化：

```python
prize = (prize_emb * prize_mask).sum(dim=2) / DECK_N
summ = summ + prize
```

这是符合规格文字的正确行为，只是 B.2 伪代码中缺少记录。

---

### 8. `vocab.json` 存储了 `w_lost`，但规格 D.3 输出列表未提及

**文件**: `python/ptcg_mine/artifacts.py:49`

`artifacts.py` 将 `"w_lost": config.w_lost` 写入 `vocab.json`。没有任何读取方从那里消费它（样本权重在 `dataset.py` 中使用编译时常量 `W_LOST=0.6` 从 `meta.parquet` 派生）。无害，但不符合规格 — `w_lost` 应属于训练配置，而非 vocab 制品。

---

## 低 — 行为正确，但存在规格歧义或风格问题

### 9. `_clip_norm` 统一裁剪到 `[-1, 1]`

规格说数量类特征裁剪到 `[0, 1]`，有符号特征裁剪到 `[-1, 1]`。代码将所有值都裁剪到 `[-1, 1]`。对于当前所有用途，数量值非负且不会超过 1，所以 `[-1, 1]` 裁剪实际上没有区别。无功能性 bug。

### 10. 标签平滑使用自定义掩码版本

**文件**: `python/ptcg_il/train/loop.py:71-116`

规格 C.6 显示 `F.cross_entropy(..., label_smoothing=0.05)`，这会将平滑质量分布到所有 O_MAX 个类别上，包括填充位置（填充位置的 logits 为 `-1e9`）。自定义的 `masked_label_smoothed_ce` 将平滑限制为仅有效选项。这是一个必要的修正 — 标准标签平滑配合 `-1e9` 填充 logits 会产生 NaN。代码正确，规格伪代码被过度简化。

### 11. MLP 辅助函数中的 Dropout

**文件**: `python/ptcg_il/model/__init__.py:9`

规格 B.1 说 `MLP(i,h,o) = Linear(i,h)→GELU→Linear(h,o)`。代码在 GELU 和第二个 Linear 之间添加了 `nn.Dropout(dropout)`。默认 `dropout=0.0`，所以它是空操作。实际 dropout 由编码器自己的 dropout 层处理。在实践中不存在差异。

### 12. 归一化常量重复定义

`featurizer.py` 和 `cards.py` 中分别定义了相同的 `HP_N=400` 等常量。目前值相同，但存在漂移风险。

### 13. QA 中的 `_build_simple_ref_from_obs` 使用错误的索引顺序

**文件**: `python/ptcg_il/qa.py:333-374`

此函数按线性顺序（手牌→板凳→出战→弃牌堆）索引实体，而非按照固定的 A.1 令牌布局。这仅用于引用往返 QA 检查（通常因缺少观察文件而被跳过），与实际的 featurizer 布局不匹配。不是生产环境的 bug，但 QA 检查如果运行会产生误报。

---

## 已验证正确的部分

以下区域已与规格核对，完全匹配：

| 区域 | 状态 |
|---|---|
| A.1 令牌布局常量（P_MAX=12, H_MAX=30, L_STATE=46, O_MAX=64） | ✅ |
| A.1 固定令牌位置（CLS=0, 宝可梦1..12, 手牌13..42, 摘要43..44, 竞技场45） | ✅ |
| A.2 归一化常量（全部 12 个值） | ✅ |
| A.3 `card_static_row`（52 维）切片布局 | ✅ |
| A.3 `attack_static_row`（14 维）切片布局 | ✅ |
| A.4 所有输出张量的键名、形状和 dtype | ✅ |
| A.5 `poke_feat`（26 维）和 `sum_feat`（11 维）布局 | ✅ |
| A.6 `cls_feat`（93 维）布局，包括选择条件 | ✅ |
| A.7 所有 OptionType（0–16）的引用解析 | ✅ |
| §1.3 错位配对规则（`range(len(steps)-1)`） | ✅ |
| B.1 `CardEncoder` / `AttackEncoder` — id 嵌入 + 静态 MLP | ✅ |
| B.2 `TokenEmbedder` — 带有加性类别嵌入的逐组投影 | ✅ |
| B.3 `Encoder` — 预归一化 Transformer，GELU | ✅ |
| B.4 `PointerHead` — 选项查询构建 + 交叉注意力 + 评分 | ✅ |
| B.5 `ValueHead` — `tanh(Linear→GELU→Linear)` | ✅ |
| B.6 `Policy.forward` — 嵌入→编码→指针+价值 | ✅ |
| B.7 多选自回归（训练时教师强制，推理时贪心） | ✅ |
| B.8 权重初始化（`trunc_normal_(std=0.02)`） | ✅ |
| C.3 样本权重公式（`w_ctx * w_arch * w_out`，均值归一化） | ✅ |
| C.5 优化器配置（AdamW，无衰减参数组，预热→余弦衰减） | ✅ |
| D.3 专家选择（按胜率排序的前 K 名，最少对局阈值） | ✅ |
| D.3 原型聚类（贪心 Jaccard 多集，阈值 0.90） | ✅ |
| D.3 𝒟_self/𝒟_opp 选择 | ✅ |
| D.4 拆分比例（训练/验证/测试 ≈ 96/2/2，按剧集哈希） | ✅ |
| D.4 `sample_weight` 未写入分片 | ✅ |
