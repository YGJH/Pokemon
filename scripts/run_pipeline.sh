#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — 完整訓練管線（五個階段）
#
#   [1/5] Rust engine    編譯 ptcg_search（MCTS + search_plan）
#   [2/5] Collect data   ptcg_mine：下載 → 統計 → archetype → vocab
#   [3/5] Build shards   特徵化成 .npz 分片（一律寫入 belief 標籤）
#   [4/5] Training       4a 每副牌一個專家模型 + 4b belief 頭 + 4c 記錄 IL 基準
#   [5/5] MCTS           AlphaZero 式自我對弈蒸餾 + league gate（ptcg_rl.mcts_train）
#
# 步驟 2、3 有指紋快取：兩者都是 (raw 語料 + 設定 + 實作它們的程式碼) 的純函數，
# 卻各要 ~15 / ~35 分鐘。輸入沒變就直接沿用既有產物。判斷寫在 ptcg_mine.mine 與
# ptcg_il.cli build-shards 裡（見 ptcg_mine/stamp.py），不在本腳本 —— 直接呼叫
# 那兩個模組也一樣會跳過。--force-mine / --force-shards / --force 可強制重跑。
#
# 有沒有 --skip-download 都會走快取：指紋是在下載「之後」才比對，所以沒抓到新
# episode 的執行不會白花 15 分鐘重算 Phase 2。--skip-download 省的是 Kaggle
# API 呼叫本身。
#
# 用法:
#   ./scripts/run_pipeline.sh                             # 全部執行
#   ./scripts/run_pipeline.sh --skip-download             # 跳過下載（已有 raw/）
#   ./scripts/run_pipeline.sh --skip-download --no-rl     # 只到訓練＋評估
#   ./scripts/run_pipeline.sh --no-train                  # 只做到建分片
#
# 階段 5 預設會跑，用 --mcts-games 控制預算、--no-rl 跳過。它比前四個階段慢得多，
# 預設的 5000000 場等於跑到手動停止（checkpoint 只在通過 gate 時寫出，停掉不會丟）。
#
# 環境要求:
#   - uv (Python 3.11+)
#   - ~/.kaggle/kaggle.json (下載需要)
# =============================================================================

set -euo pipefail

# ── 预设参数 ────────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=""
NO_TRAIN=""
NO_EVAL=""
SKIP_RUST=""
# 指紋快取的強制重跑開關（見檔頭）。預設空 = 只在指紋不符時才重算。
FORCE_MINE=""
FORCE_SHARDS=""
# 空 = 沿用 $DATA_DIR/archetypes.json 的 archetype id（append-only，舊
# checkpoint 仍然有效）。--rebaseline 會重新編號，等於作廢所有既有 checkpoint
# 的 deck 記錄與 belief slot。
REBASELINE=""
# 各阶段完整输出一律写入 logs/ 下的日志档。终端预设只印**训练与 RL**（阶段 4、5）
# 的即时输出：这两段动辄数小时，只有一行 "training..." 等于全程没有进度可看。
# download/mine/build-shards 会刷上千行、对进度没帮助，所以仍旧只留摘要。
# --verbose 连那几个阶段也印；--quiet 回到全部只印摘要（AI agent 跑管线时用，
# stdout 太吵会塞爆 context window）。
VERBOSE=""
QUIET="true"
# 评估设定。离线评估（top-1/top-3）永远跑；live-eval 要真的开对局，慢得多，
# 所以预设关闭，用 --live-eval 打开。
LIVE_EVAL=""
# belief 頭現在一律訓練（build-shards 無條件寫入標籤），所以沒有 --belief 旗標了；
# 要關掉請用 --no-belief，它會一路傳到 ptcg_il.cli。
NO_BELIEF=""
LIVE_EVAL_GAMES=200
LIVE_EVAL_WORKERS=""
# 相对路径一律以 python/ 为基准（管线全程在 python/ 下执行，
# 因为 ptcg_mine / ptcg_il 套件位于该目录，必须是 cwd 才 import 得到）。
RAW_DIR="raw"
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
N_DAYS=20
TARGET_EPISODES=5000
SEED=0
K_EXPERTS=10
G_MIN=50
JACCARD_THRESH=0.90
BATCH_SIZE=256
EPOCHS=2000

# ── 模型架構（階段 4a/4b，TRANSFORMER_IL_SPEC Appendix B）─────────────────
# 一律留空 = 沿用 ptcg_il.cli 的 DEFAULTS（d_model 256 / layers 4 / heads 8 /
# ff 1024 / dropout 0.1）。這裡刻意不寫死一份預設值：那會變成第二個真相來源，
# 改了 cli.py 而忘了改這裡，管線就會安靜地用舊架構訓練。空值 = 不傳旗標。
#
# 只影響階段 4。階段 5 不需要這些旗標：mcts_train 從 checkpoint 的
# Policy.config 讀架構（見 CLAUDE.md「Every checkpoint records its own
# architecture」），傳了反而會有兩個來源不一致的風險。
#
# 分片不受影響：build-shards 的指紋只含 featurizer 的維度（h_max/o_max/d_max），
# 不含模型架構，所以改這些不會觸發那 ~35 分鐘的重建。
IL_D_MODEL="256"
IL_LAYERS="14"
IL_HEADS="8"
IL_FF="1024"
IL_DROPOUT="0.0"
# 每个牌组训练一个专家模型（archetype specialist）。
# 六个 D_self 原型近乎互斥（pairwise multiset-Jaccard <= 0.17，没有任何一张卡同时
# 出现在全部六个里），单一通才模型必须同时拟合数个互不相干的策略，因此
# --archetype-self 的非平凡 top-1 提升是通才的 3~10 倍（arch 0: +2.7 -> +20.5；
# arch 2: +8.8 -> +49.4）。详见 CLAUDE.md「Key design decisions」。
# 每个原型输出到 ${CHECKPOINT_DIR}_a<id>/，各自带 decks.json / deck.csv 标签。
# 设为空字符串（或传 --generalist）则退回旧行为：只训练一个通才模型。
# 要訓練哪些 archetype。空字串 = 自動從 artifacts 推導（見步驟 4）。
# 千萬不要寫死 id：archetype id 只是分群索引，重跑 mine 就會重新編號。
# 本語料的 self_ids 是 [0, 1, 11, 3, 4, 5] —— 舊的預設值 "0 2" 裡的 2 根本不存在。
ARCHETYPES=""
N_ARCHETYPES=2
# mining designates 𝒟_self 的數量。步驟 4 最多只能訓練這麼多個專家模型 ——
# 要 top-10 就得先用 --n-self 10 重跑 mining（會連帶重建分片）。
N_SELF="6"

# ── 階段 5（MCTS 自我對弈）參數 ─────────────────────────────────────────────
# 階段 5 跑 AlphaZero 式的 MCTS 蒸餾（ptcg_rl.mcts_train），不是 PPO。
# 預設值一律取自 scripts/run_mcts_train.sh，兩邊要一起改。
GENERALIST=""
# 預訓練 + 微調：階段 4 先在「全部 archetype」上訓一個通才模型，階段 5 再拿它當
# θ_init，對 --rl-archetype 指定的那一副牌做 MCTS 微調。通才吃得到整個語料
# （本語料 70,375 筆，最大的專家只有 37,883 筆），專家模型則負責提供 KL anchor
# 與 league baseline —— 兩者由 mcts_train 的 --init-ckpt / --il-ckpt 分開接。
# 這是額外的一種模式，--generalist 與預設的 specialist 路徑都不受影響。
PRETRAIN_FINETUNE=""
NO_RL=""
RL_ARCHETYPE=""     # 非空 = 只對這一副牌跑，蓋過 RL_TOP_N 的排名挑選
# 階段 5 只對「表現最好的這麼多副牌」跑。一副牌動輒數小時，
# --n-archetypes all 之下全部都跑會把預算灑在明顯較弱的牌組上。
RL_TOP_N=2
RL_EXTRA=""

# MCTS 搜尋
MCTS_ITERATIONS=64          # 每棵樹的 PUCT 迭代次數
MCTS_C_PUCT=2.0             # PUCT 探索常數
MCTS_K_DET=4                # 每個 root 的 determinization 數（對手牌組已知，不必 K=8）
MCTS_LEAF_BATCH=4096        # MCTS GPU forward batch
MCTS_RHO=1.0                # ρ：多少比例的決策點要跑 MCTS
MCTS_N_ENGINES=20           # MCTS libcg 並行數
MCTS_WORKERS=16             # RolloutPool 並行對局數
MCTS_FORWARD_BATCH=4096     # rollout GPU forward batch
MCTS_ALL_ARCHETYPES=1       # 1=對全部 179 牌組訓練, 0=只用 6 個 𝒟_opp

# Replay buffer
MCTS_BUFFER_CAPACITY=10000 # 最大決策點數 — ~1.2 GB
MCTS_MIN_BUFFER=1000        # 開始訓練的最小 buffer 量

# 訓練
# 5000000 場等於「跑到你把它停掉為止」。這是刻意的：checkpoint 只在通過 league
# gate 時才寫出，中途停掉不會丟掉已通過的成果。要有限預算就調 --mcts-games。
MCTS_GAMES=5000000
MCTS_GAMES_PER_ITER=200
MCTS_TRAIN_STEPS_PER_ITER=200
MCTS_BATCH_SIZE=1024
MCTS_LR=0.00002            # fine-tuning，低於 IL 的 3e-4
MCTS_C_VALUE=0.5
MCTS_C_PI=1.0
# 30, not 0.5.  Measured over a real a1 run, raw grad norms ranged 4.87–34.93
# (median 12.68), so a 0.5 clip bound on **100%** of steps and rescaled each one
# by a different factor between 10x and 70x.  That does not merely discard
# gradient magnitude, it inverts it: every step leaves the clip at norm 0.5, so
# the batch carrying 7x more signal was divided 7x harder.  |g|=15 is also not
# large for this model — at 14.4M params it is a per-parameter RMS of 4e-3,
# while the clipped 0.5 is 1.3e-4.  Step size does not grow from this change:
# AdamW is scale-invariant in steady state (m and sqrt(v) both scale with the
# gradient), so a constant clip is nearly a no-op and `lr` still sets the step.
# Tune against train/grad_clip_frac — aim for 0.05–0.10, not 1.0.
MCTS_GRAD_CLIP=30

# League 評估
MCTS_EVAL_GAMES=100         # 每個對手的對戰場數
MCTS_EVAL_EVERY_GAMES=100   # 每多少場自我對弈評估一次
MCTS_GATE_SCORE=0.70        # 對所有對手的最低勝率門檻
MCTS_MAX_CHAMPIONS=8        # 最多保留幾個 champion（超過則淘汰最低 ELO）
MCTS_DEVICE="${DEVICE:-cuda}"

# W&B。mcts_train 的 --wandb 預設是關的，而 run_mcts_train.sh 是開的；管線跟後者
# 一致，否則從腳本搬過來會安靜地少掉所有記錄。專案獨立於 IL 的 pokemon-tcg-il：
# 兩者的 x 軸不同（optimizer step vs 自我對弈場數），放同一個專案面板讀不了。
# run 名稱按牌組加後綴，否則同名 run 只能點進 config 才分得出是哪一副。
MCTS_WANDB=1
MCTS_WANDB_PROJECT="pokemon-tcg-mcts"
MCTS_WANDB_ENTITY="poken"
# echo $1
# ── 解析参数 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --no-train)
            NO_TRAIN="true"
            shift
            ;;
        --skip-rust)
            SKIP_RUST="true"
            shift
            ;;
        --force-mine)
            FORCE_MINE="true"; shift
            ;;
        --rebaseline)
            REBASELINE="--rebaseline"; shift ;;
        --force-shards)
            FORCE_SHARDS="true"; shift
            ;;
        --force)
            FORCE_MINE="true"; FORCE_SHARDS="true"; shift
            ;;
        --raw-dir)
            RAW_DIR="$2"; shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"; shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"; shift 2
            ;;
        --n-days)
            N_DAYS="$2"; shift 2
            ;;
        --target-episodes)
            TARGET_EPISODES="$2"; shift 2
            ;;
        --seed)
            SEED="$2"; shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2
            ;;
        --epochs)
            EPOCHS="$2"; shift 2
            ;;
        --d-model)
            IL_D_MODEL="$2"; shift 2
            ;;
        --layers)
            IL_LAYERS="$2"; shift 2
            ;;
        --heads)
            IL_HEADS="$2"; shift 2
            ;;
        --ff)
            IL_FF="$2"; shift 2
            ;;
        --dropout)
            IL_DROPOUT="$2"; shift 2
            ;;
        --resume)
            RESUME_CKPT="--resume $2"; shift 2
            ;;
        --archetypes)
            ARCHETYPES="$2"; shift 2
            ;;
        --n-archetypes)
            # "all" = 每個資料量足夠的 𝒟_self 都訓練一個專家模型。實際上限是
            # mining 時 --n-self 指定的數量（預設 6），不是本語料的 179 個分群。
            if [[ "$2" == "all" ]]; then N_ARCHETYPES=999; else N_ARCHETYPES="$2"; fi
            shift 2
            ;;
        --n-self)
            N_SELF="$2"; shift 2
            ;;
        --generalist)
            GENERALIST="true"; ARCHETYPES=""; shift
            ;;
        --pretrain-finetune)
            PRETRAIN_FINETUNE="true"; shift
            ;;
        --no-eval)
            NO_EVAL="true"; shift
            ;;
        --no-belief)
            NO_BELIEF="true"; shift
            ;;
        --no-rl)
            NO_RL="true"; shift
            ;;
        --rl-archetype)
            RL_ARCHETYPE="$2"; shift 2
            ;;
        --rl-top)
            # "all" = 每個訓練好的牌組都跑（舊行為）
            if [[ "$2" == "all" ]]; then RL_TOP_N=999; else RL_TOP_N="$2"; fi
            shift 2
            ;;
        --rl-workers|--mcts-workers)
            MCTS_WORKERS="$2"; shift 2
            ;;
        --mcts-games)
            MCTS_GAMES="$2"; shift 2
            ;;
        --mcts-iterations)
            MCTS_ITERATIONS="$2"; shift 2
            ;;
        --mcts-eval-games)
            MCTS_EVAL_GAMES="$2"; shift 2
            ;;
        --mcts-gate-score)
            MCTS_GATE_SCORE="$2"; shift 2
            ;;
        --mcts-device)
            MCTS_DEVICE="$2"; shift 2
            ;;
        --mcts-resume)
            RL_EXTRA="$RL_EXTRA --resume $2"; shift 2
            ;;
        --mcts-no-wandb)
            MCTS_WANDB=0; shift
            ;;
        --mcts-wandb-project)
            MCTS_WANDB_PROJECT="$2"; shift 2
            ;;
        --live-eval)
            LIVE_EVAL="true"; shift
            ;;
        --live-eval-games)
            LIVE_EVAL_GAMES="$2"; shift 2
            ;;
        --live-eval-workers)
            LIVE_EVAL_WORKERS="$2"; shift 2
            ;;
        --verbose|-v)
            VERBOSE="true"; shift
            ;;
        --quiet|-q)
            QUIET="true"; shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "管線步驟: [1] Rust → [2] Mine → [3] Shards → [4] Train+Belief+Baselines → [5] RL"
            echo ""
            echo "路径说明: 相对路径以 python/ 为基准（预设 = python/raw, python/data,"
            echo "          python/checkpoints）。要指定别处请给绝对路径。"
            echo ""
            echo "选项:"
            echo "  --skip-download        跳过 Phase 1 下载（raw/ 已有资料时使用）"
            echo "  --skip-rust            跳过 Rust search engine 编译"
            echo ""
            echo "指紋快取 (步驟 2/3，見 python/ptcg_mine/stamp.py):"
            echo "  步驟 2 (~15 分) 與步驟 3 (~35 分) 只在輸入變了才重算。判斷寫在"
            echo "  ptcg_mine.mine 與 ptcg_il.cli build-shards 裡，直接呼叫它們也有效。"
            echo "  指紋 = raw 語料 manifest + 設定旋鈕 + 實作它們的原始碼 + 上游產物"
            echo "  sha1，記在 data/.stamp-<stage>.json。改了 featurizer 會自動重建分片。"
            echo "  下載照跑：指紋是在下載之後才比對，沒抓到新資料就不重算。"
            echo "  --force-mine           強制重跑步驟 2"
            echo "  --force-shards         強制重跑步驟 3"
            echo "  --force                兩者都強制重跑"
            echo ""
            echo "  --rebaseline           重新編號 archetype id（預設沿用既有編號）。"
            echo "                         會作廢所有既有 checkpoint 的 deck 記錄與"
            echo "                         belief slot —— --archetype-self N 會變成別副牌。"
            echo ""
            echo "  --no-train             只做 mine + build-shards，不训练"
            echo "  --raw-dir DIR          raw 目录 (默认: python/raw)"
            echo "  --data-dir DIR         产出目录 (默认: python/data)"
            echo "  --checkpoint-dir DIR   checkpoint 目录 (默认: python/checkpoints)"
            echo "  --n-days N             下载的天数 (默认: 20)"
            echo "  --target-episodes N    总 episode 目标 (默认: 500)"
            echo "  --seed N               随机种子 (默认: 0)"
            echo "  --batch-size N         训练 batch size (默认: 256)"
            echo "  --epochs N             训练 epoch 数 (默认: 500)"
            echo "  --resume PATH          从指定 checkpoint 恢复训练"
            echo ""
            echo "模型架構 (只影響階段 4；預設值來自 ptcg_il.cli，這裡不重複一份):"
            echo "  --d-model N            模型維度 D (預設: 256)"
            echo "  --layers N             Transformer encoder 層數 (預設: 4)"
            echo "  --heads N              attention head 數 (預設: 8)。D 必須能被它整除。"
            echo "  --ff N                 feed-forward 隱藏維度 (預設: 1024)"
            echo "  --dropout F            dropout (預設: 0.1)"
            echo "                         改架構不會重建分片；但新舊 checkpoint 形狀不同，"
            echo "                         --resume 舊的會失敗，階段 5 請用同一輪訓練的產物。"
            echo "  --archetypes \"0 1\"     為這些 archetype 各訓練一個專家模型"
            echo "                         (預設: 從 artifacts 自動挑資料量最大的 N 個)"
            echo "  --n-archetypes N|all   要訓練幾副牌的專家模型 (預設: 2；all = 全部"
            echo "                         資料量足夠的)。訓練時間與數量成正比。"
            echo "                         結束時會依 held-out test top-1 排名。"
            echo "  --n-self N             mining designates 幾副牌為 𝒟_self (預設 6)。"
            echo "                         要 top-10 就設 10 —— 會重跑 mining 與分片。"
            echo "  --generalist           只訓練單一通才模型 (舊行為，準確率明顯較差)"
            echo "  --no-eval              訓練後跳過評估"
            echo "  --no-belief            關閉對手牌組 belief 頭 (預設開啟；關掉會讓"
            echo "                         MCTS determinizer 退回鏡像牌組猜測)"
            echo "  --live-eval            额外跑 live engine 对局评估 (慢: 每局都要真的打完)"
            echo "  --live-eval-games N    live-eval 每个对手的局数 (默认: 200)"
            echo "  --live-eval-workers N  live-eval 并行 worker 数 (默认: 由 CLI 决定)"
            echo ""
            echo "階段 5 (MCTS 自我對弈蒸餾, ptcg_rl.mcts_train):"
            echo "  --no-rl                跳過階段 5 (比前四階段慢得多)"
            echo "  --mcts-games N         總自我對弈場數 (預設: 5000000，等同跑到手動"
            echo "                         停止；checkpoint 只在通過 gate 時寫出，"
            echo "                         中途停掉不會丟掉已通過的成果)"
            echo "  --mcts-iterations N    每棵樹的 PUCT 迭代次數 (預設: 64)"
            echo "  --mcts-eval-games N    league 評估時每個對手的場數 (預設: 100)"
            echo "  --mcts-gate-score F    存 checkpoint 所需的勝率門檻 (預設: 0.70，"
            echo "                         要對 frozen IL 與所有 past champion 都達標)"
            echo "  --mcts-workers N       RolloutPool 並行對局數 (預設: 16)"
            echo "                         (--rl-workers 是同一個旗標的舊名)"
            echo "  --mcts-device DEV      cuda / cpu (預設: \$DEVICE 或 cuda)"
            echo "  --mcts-resume FILE     從既有的 MCTS checkpoint 續跑"
            echo "  --mcts-no-wandb        關閉 W&B (預設開啟，記到 poken/pokemon-tcg-mcts，"
            echo "                         run 名稱 pokemon-tcg-mcts-a<N>)"
            echo "  --mcts-wandb-project P W&B 專案名 (預設: pokemon-tcg-mcts)"
            echo "  --rl-top N|all         只對表現最好的 N 副牌跑 (預設: 2)。"
            echo "                         排名用 4c 寫的 data/il_baselines.json"
            echo "                         (held-out test top-1)，且只挑訓練成功的。"
            echo "                         all = 每副都跑（很慢：一副要數小時）"
            echo "  --rl-archetype ID      只對這一個專家模型跑，蓋過 --rl-top"
            echo "  --pretrain-finetune    預訓練 + 微調：階段 4 訓一個吃全部語料的通才"
            echo "                         (${CHECKPOINT_DIR}_generalist) 加上目標牌組的專家，"
            echo "                         階段 5 拿通才當 θ_init、專家當 KL anchor 與"
            echo "                         league baseline 做 MCTS 微調。目標牌組取"
            echo "                         --rl-archetype，沒給就取排名第一的那副。"
            echo "                         與 --generalist / --no-rl 互斥。"
            echo ""
            echo "  --verbose, -v          所有階段的完整輸出都印到終端 (預設只印階段 4、5)"
            echo "  --quiet, -q            全部階段只印摘要，完整輸出僅寫入日誌檔"
            echo "                         (AI agent 跑管線時用，避免 context 被洗版)"
            echo "  --help, -h             顯示此幫助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"; exit 1
            ;;
    esac
done

# 旗標衝突在這裡就要擋掉。之前擺在階段 4，等於先跑完 mine + build-shards
# （本語料約 50 分鐘）才告訴使用者參數根本不能一起用。
if [[ "$PRETRAIN_FINETUNE" == "true" ]]; then
    if [[ "$GENERALIST" == "true" ]]; then
        echo "[錯誤] --pretrain-finetune 與 --generalist 互斥：後者跳過階段 5，" >&2
        echo "       前者的重點正是要跑階段 5。" >&2
        exit 1
    fi
    if [[ "$NO_RL" == "true" ]]; then
        echo "[錯誤] --pretrain-finetune 與 --no-rl 互斥：微調就是階段 5。" >&2
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY_DIR="$PROJECT_DIR/python"
RUST_DIR="$PY_DIR/ptcg_search"

# 相对路径 → 绝对路径（以 python/ 为基准），避免各步骤 cwd 不同时指到不同目录
abspath() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s' "$PY_DIR/$1" ;;
    esac
}
RAW_DIR="$(abspath "$RAW_DIR")"
DATA_DIR="$(abspath "$DATA_DIR")"
CHECKPOINT_DIR="$(abspath "$CHECKPOINT_DIR")"

cd "$PY_DIR"

# ── 日志设定 ────────────────────────────────────────────────────────────────
# 每个阶段的完整输出各写一个日志档；stdout 只印摘要。logs/latest 永远指向
# 最近一次执行的目录，方便 `tail -f logs/latest/4-train-a0.log` 盯进度。
LOG_DIR="$PY_DIR/logs/pipeline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$PY_DIR/logs/latest"

# run_stage <logfile> <cmd-string> [live]
# 输出一律进 logfile；是否同时 tee 到终端由三者决定：
#   --verbose  → 全部阶段都印
#   --quiet    → 全部阶段都不印
#   预设       → 只印 live=true 的阶段（训练与 RL）
# 训练动辄数小时，只留一行 "[4/5] training..." 等于整段没有进度可看；
# 但 mine/build-shards 会刷上千行，所以不是把 VERBOSE 直接翻成预设。
# 回传指令本身的 exit code（pipefail 之下 tee 管线也一样）。
# 少传一个参数不会报错，只会静静地做错事：`run_stage "$CMD" true` 会把整条
# 指令当成 logfile、把 "true" 当成要执行的指令，于是该阶段「成功」了却什么都
# 没跑。階段 5 就這樣空轉過。所以这里明确检查参数个数。
run_stage() {
    if [[ $# -lt 2 ]]; then
        echo "[內部錯誤] run_stage 需要 <logfile> <cmd-string> [live]，收到 $# 個參數" >&2
        return 2
    fi
    local log="$1" cmd="$2" live="${3:-false}"
    # 少传的是 logfile 时参数会整体左移，cmd 就变成 live 旗标本身——`eval true`
    # 永远成功，阶段于是「通过」却什么都没跑。指令字串不可能正好是 true/false。
    case "$cmd" in
        true|false)
            echo "[內部錯誤] run_stage 第 2 個參數是 '$cmd' —— 少傳了 logfile？" >&2
            return 2
            ;;
    esac
    # 指令本身写进 log 头，方便事后复现；只有 --verbose 才同时印到终端，
    # 免得摘要行被一条几百字的指令挤掉。
    printf '$ %s\n\n' "$cmd" > "$log"
    if [[ "$VERBOSE" == "true" ]]; then echo "  \$ $cmd"; fi
    if stage_is_live "$live"; then
        eval "$cmd" 2>&1 | tee -a "$log"
    else
        eval "$cmd" >>"$log" 2>&1
    fi
}

# stage_is_live <live>：run_stage 会不会 tee 到终端。呼叫端要先知道，
# 才能决定标签是用 `echo -n "label... "`（等 ✓ 补在同一行）还是自成一行——
# 否则阶段的第一行日志会接在 "a0... " 后面，把摘要行搅烂。
stage_is_live() {
    [[ "$VERBOSE" == "true" ]] || { [[ "$1" == "true" ]] && [[ "$QUIET" != "true" ]]; }
}

# stage_label <text> <live>：live 时自成一行，否则留在同一行等结果符号。
stage_label() {
    if stage_is_live "$2"; then echo "  $1..."; else echo -n "  $1... "; fi
}

# stage_done <text> <live> <mark>：与 stage_label 配对。
stage_done() {
    if stage_is_live "$2"; then echo "  $1 $3"; else echo "$3"; fi
}

# fail_tail <logfile> <rc>：阶段失败时印出日志末尾，让呼叫者不用开档案也能除错。
fail_tail() {
    echo "[错误] 阶段失败 (exit $2) — $1 末尾 40 行:" >&2
    tail -n 40 "$1" >&2
}

echo "=== pipeline $(date +%Y%m%d-%H%M%S) | data=$(basename "$DATA_DIR") ckpt=$(basename "$CHECKPOINT_DIR") | logs=$(basename "$LOG_DIR") ==="
[[ "$VERBOSE" == "true" ]] && echo "  dirs: raw=$RAW_DIR data=$DATA_DIR ckpt=$CHECKPOINT_DIR  (verbose on)"

# ── 步骤 1: 编译 Rust search engine ──────────────────────────────────────
echo -n "[1/5] Rust engine... "
RUST_LIB="$RUST_DIR/target/release/libptcg_search.so"
if [[ "$SKIP_RUST" == "true" ]]; then
    echo "⊘ (--skip-rust)"
elif ! command -v cargo >/dev/null 2>&1; then
    echo "⊘ (no cargo — live-eval search planner 不可用)"
else
    CARGO_LOG="$LOG_DIR/1-cargo.log"
    set +e
    run_stage "$CARGO_LOG" "cargo build --release --manifest-path '$RUST_DIR/Cargo.toml'"
    CARGO_RC=$?
    set -e
    if [[ $CARGO_RC -ne 0 ]]; then
        echo "✗"
        fail_tail "$CARGO_LOG" "$CARGO_RC"
        exit $CARGO_RC
    fi
    if [[ ! -f "$RUST_LIB" ]]; then
        echo "✗"
        echo "[错误] cargo 编译成功但未产生 $RUST_LIB" >&2
        exit 1
    fi
    echo "✓"
fi

# ── 步骤 2: Phase 0–2 语料挖掘 ────────────────────────────────────────────
# 跳過重算的判斷在 ptcg_mine.mine 裡（見 ptcg_mine/stamp.py），不在這裡：
# 直接跑 `python -m ptcg_mine.mine` 的人一樣要享受到，管線只是轉發 --force。
# 下載仍然照跑；mine 是在下載「之後」才比對指紋，所以沒抓到新 episode 的執行
# 不會白花 15 分鐘重算 Phase 2。
echo -n "[2/5] Mine... "
echo $SKIP_DOWNLOAD
MINE_CMD="uv run python -m ptcg_mine.mine \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --n-days $N_DAYS \
    --target-episodes $TARGET_EPISODES \
    --seed $SEED \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH \
    ${N_SELF:+--n-self $N_SELF --n-opp $N_SELF} \
    $REBASELINE \
    $SKIP_DOWNLOAD${FORCE_MINE:+ --force}"

if [[ -z "$SKIP_DOWNLOAD" ]] && [[ "$VERBOSE" == "true" ]]; then
    QUOTA=$(( TARGET_EPISODES / N_DAYS ))
    echo "[提示] ~$(( N_DAYS * 100 + QUOTA * N_DAYS )) 次 Kaggle API 呼叫 (${N_DAYS} 天 x ~${QUOTA}/天)；下载可续传。"
fi

MINE_LOG="$LOG_DIR/2-mine.log"
set +e
run_stage "$MINE_LOG" "$MINE_CMD"
MINE_RC=$?
set -e

if [[ $MINE_RC -eq 2 ]]; then
    echo "✗"
    echo "[错误] Phase 1 下载失败（通常是 Kaggle HTTP 429 限流）。" >&2
    echo "       已下载的会保留在 $RAW_DIR/，可续传。建议:" >&2
    echo "         $0 --n-days 5 --target-episodes 500   (小批次)" >&2
    echo "         $0 --skip-download                      (用现有资料继续)" >&2
    fail_tail "$MINE_LOG" "$MINE_RC"
    exit 2
elif [[ $MINE_RC -ne 0 ]]; then
    echo "✗"
    fail_tail "$MINE_LOG" "$MINE_RC"
    exit $MINE_RC
fi

if [[ ! -f "$DATA_DIR/vocab.json" ]]; then
    echo "✗"
    echo "[错误] mine 完成但未产生 $DATA_DIR/vocab.json — 检查 $MINE_LOG" >&2
    exit 1
fi
# 認這句完整字串，不是 "skipping" —— download 也會印 "skipping day ..."，
# 抓短字串會把「真的重算了」誤報成 cached。
if grep -q "Phase 2: artifacts up to date" "$MINE_LOG"; then
    echo "⊙ cached"
else
    echo "✓"
    # 抽出 mine 摘要的最后 6 行（避免整个 summary block 塞爆 context）
    sed -n '/=== Corpus mining summary ===/,$p' "$MINE_LOG" | tail -n 6
fi

# ── 步骤 3: Phase 3 特征化 ────────────────────────────────────────────────
# 同樣由 ptcg_il.cli build-shards 自己判斷。它的指紋含 vocab.json /
# archetypes.json 的內容 sha1，所以步驟 2 只要產出不同的 vocab，分片就一定會
# 重建 —— 卡牌 id 是 vocab 索引，錯配不會報錯。
echo -n "[3/5] Build shards... "
BS_CMD="uv run python -m ptcg_il.cli build-shards \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH${FORCE_SHARDS:+ --force}"

BS_LOG="$LOG_DIR/3-build-shards.log"
set +e
run_stage "$BS_LOG" "$BS_CMD"
BS_RC=$?
set -e
if [[ $BS_RC -ne 0 ]]; then
    echo "✗"
    fail_tail "$BS_LOG" "$BS_RC"
    exit $BS_RC
fi

if [[ ! -f "$DATA_DIR/meta.parquet" ]]; then
    echo "✗"
    echo "[错误] build-shards 完成但未产生 $DATA_DIR/meta.parquet — 检查 $BS_LOG" >&2
    exit 1
fi

SHARD_COUNT=$(ls "$DATA_DIR/shards/"*.npz 2>/dev/null | wc -l)
if grep -q "Phase 3: shards up to date" "$BS_LOG"; then
    echo "⊙ cached ($SHARD_COUNT shards)"
else
    echo "✓ ($SHARD_COUNT shards)"
    grep "Shard build complete" "$BS_LOG" || true
fi

# ── 步骤 4: 训练 ──────────────────────────────────────────────────────────
if [[ "$NO_TRAIN" == "true" ]]; then
    echo "═══ --no-train 已指定，跳过训练 ═══"
    echo "管线结束。产物在: $DATA_DIR/"
    exit 0
fi

echo "[4/5] Training (4a specialists + 4b belief heads)"

# 前置检查：训练需要 torch
if ! uv run python -c "import torch" >/dev/null 2>&1; then
    echo "✗"
    echo "[错误] 找不到 torch — 执行 'uv sync' 安装。" >&2
    echo "       之后可手动训练: cd $PY_DIR && uv run python -m ptcg_il.cli train --data-dir $DATA_DIR --out-dir $CHECKPOINT_DIR" >&2
    exit 1
fi

# 架構旗標的前置檢查。nn.MultiheadAttention 要求 D % heads == 0，違反時會在
# torch 深處才炸，而那時候管線已經跑完 mine 與分片了。只設其中一個也要檢查，
# 所以另一個的預設值從 ptcg_il.cli 讀 —— 不在這裡再抄一份。
if [[ -n "$IL_D_MODEL$IL_HEADS" ]]; then
    set +e
    uv run python - "$IL_D_MODEL" "$IL_HEADS" <<'PY'
import sys
from ptcg_il.cli import DEFAULTS
d = int(sys.argv[1]) if sys.argv[1] else DEFAULTS["d_model"]
h = int(sys.argv[2]) if sys.argv[2] else DEFAULTS["heads"]
if d % h:
    sys.exit(f"[錯誤] --d-model {d} 不能被 --heads {h} 整除 "
             f"(nn.MultiheadAttention 的硬性要求)")
print(f"  arch: d_model={d} heads={h}")
PY
    ARCH_CHECK_RC=$?
    set -e
    if [[ $ARCH_CHECK_RC -ne 0 ]]; then exit $ARCH_CHECK_RC; fi
fi

build_train_cmd() {
    local out_dir="$1" arch="$2"
    local cmd="uv run python -m ptcg_il.cli train \
    --data-dir $DATA_DIR \
    --out-dir $out_dir \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS"
    # 只在使用者真的指定時才傳 —— 沒傳就讓 ptcg_il.cli 用自己的預設。
    if [[ -n "$IL_D_MODEL" ]]; then cmd="$cmd --d-model $IL_D_MODEL"; fi
    if [[ -n "$IL_LAYERS"  ]]; then cmd="$cmd --layers $IL_LAYERS"; fi
    if [[ -n "$IL_HEADS"   ]]; then cmd="$cmd --heads $IL_HEADS"; fi
    if [[ -n "$IL_FF"      ]]; then cmd="$cmd --ff $IL_FF"; fi
    if [[ -n "$IL_DROPOUT" ]]; then cmd="$cmd --dropout $IL_DROPOUT"; fi
    if [[ -n "$arch" ]]; then
        cmd="$cmd --archetype-self $arch"
    fi
    if [[ "$NO_BELIEF" == "true" ]]; then
        cmd="$cmd --no-belief"
    fi
    if [[ -n "${RESUME_CKPT:-}" ]]; then
        cmd="$cmd $RESUME_CKPT"
    fi
    printf '%s' "$cmd"
}

# archetype id 是分群索引，重跑 mine 就會重新編號 —— 所以從 artifacts 推導，
# 不寫死。GENERALIST 用空的 ARCHETYPES 表示，跟「還沒推導」區分開來。
if [[ "$GENERALIST" != "true" ]] && [[ -z "${ARCHETYPES// /}" ]]; then
    echo -n "  archetypes: "
    set +e
    ARCHETYPES=$(uv run python -m ptcg_il.cli archetypes \
        --data-dir "$DATA_DIR" --top "$N_ARCHETYPES" 2>"$LOG_DIR/4-archetypes.log")
    ARCH_RC=$?
    set -e
    if [[ $ARCH_RC -ne 0 ]] || [[ -z "${ARCHETYPES// /}" ]]; then
        echo "✗"
        fail_tail "$LOG_DIR/4-archetypes.log" "$ARCH_RC"
        echo "[提示] 用 'uv run python -m ptcg_il.cli archetypes --data-dir $DATA_DIR --describe' 看各 archetype 的資料量" >&2
        exit 1
    fi
    echo "$ARCHETYPES"
fi

# ── 預訓練 + 微調模式 ────────────────────────────────────────────────────
# 階段 4 訓兩個模型：一個吃全部語料的通才（θ_init），一個微調目標牌組的專家
# （KL anchor + league baseline）。階段 5 只跑那一副牌，用 --init-ckpt 把兩者接
# 起來。要兩個模型是因為「這樣做有沒有比原本好」只有拿專家當 baseline 才答得出來。
if [[ "$PRETRAIN_FINETUNE" == "true" ]]; then
    # --generalist / --no-rl 的衝突已在參數解析後擋掉。
    # 沒指定就取排名第一的那副。archetype id 是分群索引，不寫死。
    if [[ -z "${RL_ARCHETYPE// /}" ]]; then
        RL_ARCHETYPE=$(echo "$ARCHETYPES" | tr ' ' '\n' | head -n 1)
        echo "  finetune target: a${RL_ARCHETYPE} (未指定 --rl-archetype，取排名第一)"
    else
        # 指定的那副也得真的訓練得起來，否則階段 5 會找不到 ckpt-best.pt。
        if ! echo " $ARCHETYPES " | grep -q " $RL_ARCHETYPE "; then
            echo "[錯誤] --rl-archetype $RL_ARCHETYPE 不在可訓練的 archetype 列表內 ($ARCHETYPES)。" >&2
            echo "       用 'uv run python -m ptcg_il.cli archetypes --data-dir $DATA_DIR --describe' 看哪些 usable。" >&2
            exit 1
        fi
        echo "  finetune target: a${RL_ARCHETYPE}"
    fi
    # 階段 4 只訓這一副的專家 —— 其他副在這個模式下沒有用途，訓了只是白花時間。
    ARCHETYPES="$RL_ARCHETYPE"
fi

# 收集要训练的 (输出目录, archetype) 组合
TRAIN_TARGETS=()
if [[ "$PRETRAIN_FINETUNE" == "true" ]]; then
    # 通才排前面：階段 5 需要它當 θ_init，先訓好才不會白跑一輪專家。
    TRAIN_TARGETS+=("${CHECKPOINT_DIR}_generalist|")
    TRAIN_TARGETS+=("${CHECKPOINT_DIR}_a${RL_ARCHETYPE}|${RL_ARCHETYPE}")
elif [[ -z "${ARCHETYPES// /}" ]]; then
    TRAIN_TARGETS+=("$CHECKPOINT_DIR|")
else
    for arch in $ARCHETYPES; do
        TRAIN_TARGETS+=("${CHECKPOINT_DIR}_a${arch}|${arch}")
    done
fi

echo "  ${#TRAIN_TARGETS[@]} model(s)"

TRAINED_DIRS=()
for target in "${TRAIN_TARGETS[@]}"; do
    out_dir="${target%%|*}"
    arch="${target##*|}"
    TRAIN_CMD="$(build_train_cmd "$out_dir" "$arch")"
    TRAIN_LOG="$LOG_DIR/4-train${arch:+-a$arch}.log"

    LABEL="${arch:+a$arch}"
    LABEL="${LABEL:-generalist}"
    stage_label "$LABEL" true
    set +e
    run_stage "$TRAIN_LOG" "$TRAIN_CMD" true
    TRAIN_RC=$?
    set -e
    if [[ $TRAIN_RC -ne 0 ]]; then
        stage_done "$LABEL" true "✗"
        fail_tail "$TRAIN_LOG" "$TRAIN_RC"
        exit $TRAIN_RC
    fi
    stage_done "$LABEL" true "✓"
    # 训练摘要：最后几次 eval + 完成行
    grep -E "eval step|Training complete" "$TRAIN_LOG" | tail -n 3 || true
    TRAINED_DIRS+=("$out_dir")
done

# ELO 校正跑在「即將進階段 5 的那副牌」上。之前是無參數呼叫，而
# elo_calibrate.sh 的預設是 a0 —— 本語料的 self_ids 是 [1, 17, 25, 16, 36, 21]，
# 根本沒有 a0，於是它 exit 1，而這裡在 `set -e` 底下，整條管線就死在階段 5 前面。
# archetype id 是分群索引，不能寫死；沿用上面已經推導出來的那個。
ELO_ARCH=""
if [[ -n "${RL_ARCHETYPE// /}" ]]; then
    ELO_ARCH="a${RL_ARCHETYPE}"
elif [[ -n "${ARCHETYPES// /}" ]]; then
    ELO_ARCH="a$(echo "$ARCHETYPES" | tr ' ' '\n' | head -n 1)"
fi
if [[ -z "$ELO_ARCH" ]]; then
    # 純通才模式沒有「某一副牌」可校正。
    echo "  elo: ⊘ (generalist — 沒有對應的 checkpoints_a<N>)"
elif [[ ! -d "$PY_DIR/${CHECKPOINT_DIR}_${ELO_ARCH}" ]]; then
    echo "  elo: ⊘ (找不到 ${CHECKPOINT_DIR}_${ELO_ARCH})"
else
    cd "$PROJECT_DIR"
    # 校正失敗不該讓已經訓練好的模型跟著陪葬 —— 它只是排名，不是產物。
    set +e
    ./scripts/elo_calibrate.sh "$ELO_ARCH"
    ELO_RC=$?
    set -e
    if [[ $ELO_RC -ne 0 ]]; then
        echo "  elo: ✗ (exit $ELO_RC) — 繼續跑階段 5" >&2
    fi
    cd "$PY_DIR"
fi

# ── 步驟 5: MCTS 自我對弈蒸餾 (AlphaZero 式) ───────────────────────────────
# 對每個訓練好的專家模型各跑一次，依序而非同時。RL_SPEC §10.1 決定用交替
# (alternating) 而非併行訓練：同時訓練兩個模型會讓雙方都變成非穩態，任何回歸
# 都變成耦合動力學問題，go/no-go 也就無法歸因。
#
# 這裡跑的是 ptcg_rl.mcts_train，不是 R1/R2 的 PPO 路徑：自我對弈產生 (π̃, Ṽ)
# 目標，policy 蒸餾到那個目標上，checkpoint 只在通過 league gate（對 frozen IL
# 與所有 past champion 勝率都 ≥ MCTS_GATE_SCORE）時才寫出，最多留
# MCTS_MAX_CHAMPIONS 個。參數與 scripts/run_mcts_train.sh 同源。
#
# --deck-archetype 一定要跟 --il-ckpt 同一個 N：archetypes.json 的 fixed_deck
# 只是「最好的那個 𝒟_self 原型」的代表牌組（本語料是 archetype 4），拿它去配
# 別的專家模型，模型會收到自己從沒見過的卡 —— 那些卡如今是全零特徵列，不會
# 報錯。mcts_train 會比對 checkpoint 上蓋的 deck 記錄，不符就直接中止。

echo "═══ pipeline 完成，產物在: $DATA_DIR/ ═══ 不想跑階段 5 可用 --no-rl ═══"

RL_DIRS=()
RL_FAILED=()
if [[ "$NO_RL" == "true" ]]; then
    echo "═══ --no-rl — 跳過階段 5 ═══"
elif [[ "${#TRAINED_DIRS[@]}" -eq 0 ]]; then
    echo "[5/5] RL... ⊘ (沒有訓練好的模型)"
elif [[ -z "${ARCHETYPES// /}" ]]; then
    # 通才模型沒有固定牌組，但 RL_SPEC 整套設計都建立在「每個模型擁有一副
    # 標籤釘死的牌組」之上（§10.1）。所以跳過，而不是硬湊一個 archetype id。
    echo "[5/5] RL... ⊘ (generalist 模型沒有固定牌組；RL 需要 --archetypes)"
else
    # 挑哪幾副牌進 RL。
    #
    # 預設只取「表現最好的 RL_TOP_N 副」，而不是每一副都跑：一副牌就要
    # 數小時，--n-archetypes all 之下全跑等於把預算平均灑在明顯較弱的牌組上。
    # 排名用 4c 寫進 data/il_baselines.json 的 held-out test 分數，並且只在
    # 真的訓練成功（有 ckpt-best.pt）的牌組裡挑。
    #
    # 牌組會跟著模型走：--il-ckpt 指向 checkpoints_a<N>/，--deck-archetype 也是
    # 同一個 N，mcts_train 再用它從 data/ 重建牌表，三者同源。
    if [[ -n "$RL_ARCHETYPE" ]]; then
        RL_TARGETS="$RL_ARCHETYPE"
        RL_PICK="--rl-archetype"
    else
        RL_TARGETS=""
        if [[ -f "$DATA_DIR/il_baselines.json" ]]; then
            RL_TARGETS=$(uv run python - "$DATA_DIR/il_baselines.json" "$RL_TOP_N" \
                             "$CHECKPOINT_DIR" <<'PY' 2>"$LOG_DIR/5-rl-select.log" || true
import json, sys
from pathlib import Path
rows = json.load(open(sys.argv[1]))
top_n, ckpt_root = int(sys.argv[2]), sys.argv[3]
scored = []
for arch, rec in rows.items():
    score = rec.get("top1_macro")
    if score is None:
        continue
    if not Path(f"{ckpt_root}_a{arch}/ckpt-best.pt").is_file():
        continue
    scored.append((float(score), str(arch)))
scored.sort(key=lambda r: -r[0])
print(" ".join(a for _, a in scored[:top_n]))
PY
                        )
            RL_PICK="best $RL_TOP_N by held-out test top1"
        fi
        if [[ -z "${RL_TARGETS// /}" ]]; then
            # 沒有 baselines（例如 --no-eval）就退回訓練順序，它已經照資料量
            # 排過。講明白是退回，不要讓人以為這是照分數挑的。
            RL_TARGETS=$(echo "$ARCHETYPES" | tr ' ' '\n' | head -n "$RL_TOP_N" | tr '\n' ' ')
            RL_PICK="no il_baselines.json — falling back to training order"
        fi
    fi

    RL_N=$(echo "$RL_TARGETS" | wc -w)
    echo "[5/5] MCTS: $RL_N deck(s) — $RL_TARGETS  [$RL_PICK]"

    for rl_arch in $RL_TARGETS; do
        RL_CKPT_DIR="${CHECKPOINT_DIR}_a${rl_arch}"
        RL_IL_CKPT="$RL_CKPT_DIR/ckpt-best.pt"
        RL_OUT_DIR="${RL_CKPT_DIR}_mcts"

        RL_LOG="$LOG_DIR/5-mcts-a${rl_arch}.log"

        stage_label "a${rl_arch}" true

        if [[ ! -f "$RL_IL_CKPT" ]]; then
            stage_done "a${rl_arch}" true "✗"
            echo "[錯誤] 找不到 $RL_IL_CKPT —— 階段 5 需要一個訓練好的專家模型" >&2
            RL_FAILED+=("a${rl_arch} (缺 ckpt-best.pt)")
            continue
        fi

        # 預訓練 + 微調：θ_init 換成通才，--il-ckpt 仍是專家，繼續當 KL anchor
        # 與 league baseline。gate 問的因此是「通才微調過後有沒有贏過專家」——
        # 兩邊都指向通才的話這題就問不出來了。
        RL_INIT=""
        if [[ "$PRETRAIN_FINETUNE" == "true" ]]; then
            RL_GENERALIST_CKPT="${CHECKPOINT_DIR}_generalist/ckpt-best.pt"
            if [[ ! -f "$RL_GENERALIST_CKPT" ]]; then
                stage_done "a${rl_arch}" true "✗"
                echo "[錯誤] 找不到 $RL_GENERALIST_CKPT —— --pretrain-finetune 需要階段 4 的通才模型" >&2
                RL_FAILED+=("a${rl_arch} (缺 generalist ckpt-best.pt)")
                continue
            fi
            RL_INIT=" --init-ckpt $RL_GENERALIST_CKPT"
        fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # --init-ckpt 只由 $RL_INIT 提供。這裡本來還額外寫死一行
        # `--init-ckpt checkpoints_generalist/ckpt-best.pt`，於是
        # --pretrain-finetune 之下同一個旗標出現兩次 —— argparse 取最後一個，
        # 不會報錯；而在非 --pretrain-finetune 之下，那個寫死的路徑根本不保證
        # 存在，卻讓每一次階段 5 都偷偷從通才起跑。
        RL_CMD="uv run python -m ptcg_rl.mcts_train \
    --data-dir $DATA_DIR \
    --il-ckpt $RL_IL_CKPT$RL_INIT \
    --out-dir $RL_OUT_DIR \
    --deck-archetype $rl_arch \
    --iterations $MCTS_ITERATIONS \
    --c-puct $MCTS_C_PUCT \
    --k-determinizations $MCTS_K_DET \
    --leaf-batch $MCTS_LEAF_BATCH \
    --rho $MCTS_RHO \
    --mcts-distill \
    --n-engines $MCTS_N_ENGINES \
    --n-workers $MCTS_WORKERS \
    --forward-batch $MCTS_FORWARD_BATCH \
    --buffer-capacity $MCTS_BUFFER_CAPACITY \
    --min-buffer $MCTS_MIN_BUFFER \
    --total-games $MCTS_GAMES \
    --games-per-iter $MCTS_GAMES_PER_ITER \
    --train-steps-per-iter $MCTS_TRAIN_STEPS_PER_ITER \
    --batch-size $MCTS_BATCH_SIZE \
    --lr $MCTS_LR \
    --c-value $MCTS_C_VALUE \
    --c-pi $MCTS_C_PI \
    --grad-clip $MCTS_GRAD_CLIP \
    $([ "$MCTS_ALL_ARCHETYPES" = "1" ] && echo "--all-archetypes") \
    --eval-games $MCTS_EVAL_GAMES \
    --eval-every-games $MCTS_EVAL_EVERY_GAMES \
    --gate-score $MCTS_GATE_SCORE \
    --max-champions $MCTS_MAX_CHAMPIONS \
    --seed $SEED \
    --device $MCTS_DEVICE \
    $([ "$MCTS_WANDB" = "1" ] \
        && echo "--wandb --wandb-project $MCTS_WANDB_PROJECT --wandb-entity $MCTS_WANDB_ENTITY --wandb-name ${MCTS_WANDB_PROJECT}-a${rl_arch}" \
        || echo "--no-wandb")$RL_EXTRA"

        set +e
        run_stage "$RL_LOG" "$RL_CMD" true
        RL_RC=$?
        set -e
        # 之前这里无条件打 ✓ 并把 out-dir 记进 RL_DIRS —— 阶段 5 整个没跑成
        # 也照样「成功」，摘要还列出一个空目录。exit code 才是唯一凭据。
        if [[ $RL_RC -ne 0 ]]; then
            stage_done "a${rl_arch}" true "✗"
            fail_tail "$RL_LOG" "$RL_RC"
            RL_FAILED+=("a${rl_arch} (exit $RL_RC)")
            continue
        fi
        stage_done "a${rl_arch}" true "✓"
        RL_DIRS+=("$RL_OUT_DIR")
    done
fi

echo ""
echo "=== done ==="

# 訓練了多副牌時，排出名次 —— 否則要自己開 il_baselines.json 比對。
# 這是 held-out test 的離線 top-1，不是勝率：要決定「提交哪一副」請再對排前面
# 的幾個跑 --live-eval，離線準確率高不等於實戰贏得多。
if [[ "${#TRAINED_DIRS[@]}" -gt 1 ]] && [[ -f "$DATA_DIR/il_baselines.json" ]]; then
    echo "  ranking (held-out test top1_macro):"
    uv run python - "$DATA_DIR/il_baselines.json" <<'PY' || true
import json, sys
rows = json.load(open(sys.argv[1]))
rank = sorted(((str(k), v.get("top1_macro")) for k, v in rows.items()
               if v.get("top1_macro") is not None), key=lambda r: -r[1])
for i, (arch, score) in enumerate(rank, 1):
    print(f"    {i}. archetype {arch:<3} {score:.4f}"
          + ("   <- best offline" if i == 1 else ""))
PY
fi

for d in "${TRAINED_DIRS[@]}"; do
    if [[ -f "$d/ckpt-best.pt" ]]; then
        echo "  IL model: $d/ckpt-best.pt"
    else
        echo "  IL model: $d/ckpt-best.pt  [不存在]"
    fi
done
for d in "${RL_DIRS[@]}"; do
    # champion checkpoint 只在通過 gate 時才寫，所以列實際存在的那些，
    # 而不是印一個固定檔名讓人以為一定有。
    latest=$(ls -1 "$d"/ckpt-mcts-champion-*.pt 2>/dev/null | tail -n 1)
    if [[ -n "$latest" ]]; then
        echo "  MCTS champion: $latest"
    else
        echo "  MCTS champion: [沒有 champion 通過 gate — 別拿 last 當成果]"
    fi
    [[ -f "$d/ckpt-mcts-last.pt" ]] && echo "  MCTS last:     $d/ckpt-mcts-last.pt"
    [[ -f "$d/elo_ratings.json" ]] && echo "  MCTS ELO:      $d/elo_ratings.json"
done
echo "  data:  $DATA_DIR/"
if [[ -f "$DATA_DIR/il_baselines.json" ]]; then
    echo "  baselines: $DATA_DIR/il_baselines.json"
fi
echo "  logs:  $LOG_DIR/"

if [[ "${#EVAL_FAILED[@]}" -gt 0 ]]; then
    echo ""
    echo "[警告] 評估/基準失敗或跳過:" >&2
    for f in "${EVAL_FAILED[@]}"; do
        echo "         - $f" >&2
    done
fi

if [[ "${#RL_FAILED[@]}" -gt 0 ]]; then
    echo ""
    echo "[錯誤] RL 失敗:" >&2
    for f in "${RL_FAILED[@]}"; do
        echo "         - $f" >&2
    done
fi

if [[ "${#EVAL_FAILED[@]}" -gt 0 ]] || [[ "${#RL_FAILED[@]}" -gt 0 ]]; then
    exit 1
fi
