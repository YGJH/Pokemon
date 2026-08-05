# 集成策略设计 — 流程与逻辑

**日期**：2026-08-05
**状态**：已批准

## 我们要构建什么

用不同的随机种子训练 N 个相同牌组专精模型，推理时对它们的预测取平均。成员看到完全相同的数据（同一个牌组原型、同一个训练/验证/测试分割），但权重初始化和批次顺序不同 — 这足以去相关它们的误差。对输出概率取平均能得到比任何单一成员更好的策略。

集成仅支持贪婪模式（不含 MCTS）。每个决策的成本是 N 次前向传递，而不是 1 次。

---

## 构建顺序

```
A (--seed) → C (AR 循环重构) → B (EnsemblePolicy 模块) → E (测量) → D (打包)
```

**A、C、B** 是基础设施。**E** 在投入 GPU 时间训练新种子之前，先在已有 checkpoint 上验证基础设施是否正确。**D** 只在实时测量证明集成确实能赢时才执行。

---

## A. 训练器的 --seed

给训练 CLI 加上 `--seed INT`（默认 42），做两件事：

1. 模型构建前调用 `torch.manual_seed(seed)` — 不同种子产生不同的初始权重。
2. 控制数据集的 shuffle 顺序 — 不同种子产生不同的批次序列。

当 dropout 为 0 时，这是唯一的去相关来源。种子会被记录到 `Policy.config` 中，这样 checkpoint 就知道是哪个种子产生的。

**为什么安全**：训练/验证/测试分割来自 shard 文件名前缀，不依赖 RNG。改变种子不会改变数据属于哪个分割，所以成员的评估指标保持可比。

训练脚本 `scripts/run_ensemble_train.sh a1 3` 用 `--seed 0`、`--seed 1`、`--seed 2` 运行同一个训练命令三次，输出到不同的目录。

---

## B. EnsemblePolicy 模块

新文件：`python/ptcg_il/ensemble.py`。

一个包裹了 N 个成员 `Policy` 的 `nn.Module`。通过类方法构造，自动检测 EMA 权重和原始权重格式，从每个 checkpoint 的 config 重建成员，并验证所有成员共享相同的词汇表、原型表和牌组。

**异构成员是允许的。** 每个成员独立编码，所以不同的深度/宽度不会混淆。

### 单选择推理：forward

每个成员产生 logits → 屏蔽无效选项 → softmax → 概率。跨成员平均概率，然后取 log。结果可以直接替换现有所有做 `.masked_fill(...).argmax()` 的地方。

价值预测（游戏结果）取成员平均。贪婪模式下不使用，只在 MCTS 中有意义。

### 多选择推理：select_multi

每个成员有自己独立的多选 GRU 状态。在每个自回归步骤：

1. 每个成员为下一个选择产生 logits。
2. 每个成员自己 softmax → 概率。
3. 跨成员平均概率 → 一个 argmax 决定选项。
4. 每个成员用**自己的**选项表征在**集成选择**的索引处更新自己的 GRU。

关键点：成员以联合决策为条件 — 这就是集成，而不是 N 个独立代理并行运行。

### 调度机制

模块级的 `select_multi(policy, x)` 函数在开头加一行检查：如果对象有自己定义的 `select_multi` 方法，就委托给它。`Policy` 没有定义这个方法（它是模块级函数，不是方法），所以只有 `EnsemblePolicy` 会触发这条路径。

### 信念头怎么办

委托给第一个成员。信念只被 MCTS 消费，而集成是纯贪婪的。以后如果需要 MCTS+集成，再加。

---

## C. AR 循环重构

底层的 `_select_multi_raw` 函数目前只认一个 pointer head 和一个 GRU 状态。改为可选地接受 pointer 列表和编码状态列表。

- 两个参数都是 `None`：行为与今天逐字节相同。现有一条路径不受影响。
- 传入列表时：迭代 N 个 pointer/状态对，每一步对它们的 softmax 输出取平均，推进 N 个独立的 GRU 状态。

回归测试验证 `pointers=[policy.pointer]`（单元素列表）产生与旧单 pointer 路径完全相同的输出 — 确保重构不改变现有行为。

---

## D. 打包

### build_submission.py

`--ckpt` 变成可重复的。一个路径 = 今天的输出，逐字节相同。

两个及以上路径触发集成模式：
- 隐式 `--no-mcts`（集成仅贪婪）。
- 每个 checkpoint 的工件 SHA 与 `--data-dir` 交叉检查。词汇表不匹配警告；成员间牌组不匹配致命。
- 权重写到 `data/model_0.pt` … `data/model_{N-1}.pt`。
- 最小化的 `data/ensemble.json` 清单。
- `ensemble.py` 加入模型包，附带导入改写规则。
- 输出自动命名为 `submission-greedy-ensN.tar.gz`。

### main.py 模板

贪婪模板新增条件分支：如果 `ensemble.json` 存在，从清单构建 `EnsemblePolicy`；否则像今天一样构建单个 `Policy`。此后 `_model` 变量满足相同接口，`agent()` 的其余部分不变。

### build_submit.sh

新增 `--ensemble` 标志，接受 glob 或显式路径。脚本展开 glob 并传给 `build_submission.py`。

```bash
./scripts/build_submit.sh a1 --ensemble "python/checkpoints_a1_s*/ckpt-best.pt"
```

---

## E. 测量

### 离线评估

`--eval-only` 带多个 `--ckpt` 触发集成评估模式：

1. 构建每个成员和集成。
2. **在验证集上**：评估每个成员 + 集成。打印比较表。
3. **在测试集上**：只评估集成和验证集上最好的单个成员。保留测试集作为最终对比。

无需新的 CLI 标志 — 重复 `--ckpt` 就是信号。

### 实时评估

把 `EnsemblePolicy` 放进 `PolicyAgent`（它已经接受任何有 `forward` + `select_multi` 的对象）。跑面对面博弈：集成 vs. 最佳单成员。

### 出货规则

**集成只有在实时博弈中对最佳单成员的胜率 Wilson 下界超过 50% 时才能出货。** 离线 top-1 不做这个决定 — 这个项目已经存在 78% 离线 / 48% 实时的差距。

### 扩展循环

从 3 个成员开始。如果集成通过 Wilson 门槛，训练第 4、第 5 个成员，当实时胜率曲线趋于平坦时停止。成本：每个决策 N 次前向传递，当前模型大小约每次 1-2 ms。3 个成员约 3-6 ms 每个决策 — 在 Kaggle 时间限制内。

---

## 涉及的文件

| 文件 | 改动 |
|------|------|
| `python/ptcg_il/cli.py` | `--seed` 标志；`--eval-only` 中的集成评估 |
| `python/ptcg_il/train/loop.py` | `seed` 参数替换硬编码的 42 |
| `python/ptcg_il/model/policy.py` | config 记录 seed；`_select_multi_raw` 支持多 pointer 模式；`select_multi` 调度 |
| `python/ptcg_il/ensemble.py` | **新文件** — `EnsemblePolicy` |
| `python/tests/test_model_policy.py` | 重构后 AR 循环的单成员回归测试 |
| `python/tests/test_ensemble.py` | **新文件** — 集成构建、forward、select_multi |
| `scripts/build_submission.py` | `--ckpt` 可重复；多 checkpoint 打包；集成清单 |
| `scripts/build_submit.sh` | `--ensemble` 标志 |
| `scripts/run_ensemble_train.sh` | **新文件** — 训练循环 |
