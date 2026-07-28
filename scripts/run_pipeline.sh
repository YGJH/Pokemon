#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — 完整訓練管線（五個階段）
#
#   [1/5] Rust engine    編譯 ptcg_search（MCTS + search_plan）
#   [2/5] Collect data   ptcg_mine：下載 → 統計 → archetype → vocab
#   [3/5] Build shards   特徵化成 .npz 分片（一律寫入 belief 標籤）
#   [4/5] Training       4a 每副牌一個專家模型 + 4b belief 頭 + 4c 記錄 IL 基準
#   [5/5] RL             5a R1 critic 修復 → 5b R2 PPO + KL anchor（見 RL_SPEC.md）
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
# 階段 5 預設會跑，用 --rl-steps 控制預算、--no-rl 跳過。它比前四個階段慢得多。
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
EPOCHS=200
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

# ── 階段 5（RL）參數 ────────────────────────────────────────────────────────
GENERALIST=""
NO_RL=""
RL_STEPS=50000
RL_CRITIC_STEPS=5000
RL_ARCHETYPE=""     # 非空 = 只對這一副牌跑 RL，蓋過 RL_TOP_N 的排名挑選
# 階段 5 只對「表現最好的這麼多副牌」跑。一副牌的 R1+R2 要數小時，
# --n-archetypes all 之下全部都跑會把預算灑在明顯較弱的牌組上。
RL_TOP_N=2
RL_WORKERS=6
RL_GATE_GAMES=400
RL_PHASE="all"
RL_EXTRA=""
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
        --no-eval)
            NO_EVAL="true"; shift
            ;;
        --no-belief)
            NO_BELIEF="true"; shift
            ;;
        --no-rl)
            NO_RL="true"; shift
            ;;
        --rl-steps)
            RL_STEPS="$2"; shift 2
            ;;
        --rl-critic-steps)
            RL_CRITIC_STEPS="$2"; shift 2
            ;;
        --rl-archetype)
            RL_ARCHETYPE="$2"; shift 2
            ;;
        --rl-top)
            # "all" = 每個訓練好的牌組都跑 RL（舊行為）
            if [[ "$2" == "all" ]]; then RL_TOP_N=999; else RL_TOP_N="$2"; fi
            shift 2
            ;;
        --rl-workers)
            RL_WORKERS="$2"; shift 2
            ;;
        --rl-gate-games)
            RL_GATE_GAMES="$2"; shift 2
            ;;
        --rl-phase)
            RL_PHASE="$2"; shift 2
            ;;
        --rl-force)
            RL_EXTRA="$RL_EXTRA --force"; shift
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
            echo "階段 5 (RL, 見 RL_SPEC.md):"
            echo "  --no-rl                跳過階段 5 (RL 比前四階段慢得多)"
            echo "  --rl-steps N           PPO 優化步數預算 (預設: 50000)"
            echo "  --rl-critic-steps N    R1 critic 修復步數 (預設: 5000)"
            echo "  --rl-top N|all         只對表現最好的 N 副牌跑 RL (預設: 2)。"
            echo "                         排名用 4c 寫的 data/il_baselines.json"
            echo "                         (held-out test top-1)，且只挑訓練成功的。"
            echo "                         all = 每副都跑（很慢：一副 R1+R2 要數小時）"
            echo "  --rl-archetype ID      只對這一個專家模型跑 RL，蓋過 --rl-top"
            echo "  --rl-workers N         rollout worker 行程數 (預設: 6，見 RL_SPEC §6.5)"
            echo "  --rl-gate-games N      gate 的成對對局數 (預設: 400)"
            echo "  --rl-phase r1|r2|all   只跑 R1、只跑 R2、或兩者 (預設: all)"
            echo "  --rl-force             R1 gate 沒過也硬跑 R2 (只供除錯：critic 壞掉時"
            echo "                         PPO 等同無變異數縮減的 REINFORCE)"
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
run_stage() {
    local log="$1" cmd="$2" live="${3:-false}"
    echo command "$cmd"\n
    if stage_is_live "$live"; then
        eval "$cmd" 2>&1 | tee "$log"
    else
        eval "$cmd" >"$log" 2>&1
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

build_train_cmd() {
    local out_dir="$1" arch="$2"
    local cmd="uv run python -m ptcg_il.cli train \
    --data-dir $DATA_DIR \
    --out-dir $out_dir \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS"
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

# 收集要训练的 (输出目录, archetype) 组合
TRAIN_TARGETS=()
if [[ -z "${ARCHETYPES// /}" ]]; then
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

# ── 步驟 4c: 評估 + 記錄 IL 基準 ────────────────────────────────────────────
# 兩件事一起做，因為它們是同一次 offline eval：
#   * 一般評估跑 val split（跟訓練期間的 model selection 一致）
#   * 基準跑 test split，並把分數寫進 data/il_baselines.json，用 SHA-1 綁定
#     checkpoint。RL 的 gate 讀這個檔；SHA 對不上就拒絕執行，而不是拿別的模型
#     的分數來比。RL_SPEC §13 把「過期的 IL 基準」列為高風險且無聲的失敗。
EVAL_FAILED=()
if [[ "$NO_EVAL" == "true" ]]; then
    echo "═══ --no-eval — 跳過評估與基準記錄 ═══"
    echo "[提示] 沒有 il_baselines.json，階段 5 的 IL 回歸檢查會被跳過。" >&2
else
    echo -n "[4c/5] Eval + baselines... "

    build_eval_cmd() {
        local out_dir="$1" arch="$2" split="$3" record="$4"
        local cmd="uv run python -m ptcg_il.cli train \
    --eval-only \
    --data-dir $DATA_DIR \
    --out-dir $out_dir \
    --resume $out_dir/ckpt-best.pt \
    --eval-split $split"
        if [[ -n "$arch" ]]; then
            cmd="$cmd --archetype-self $arch"
        fi
        if [[ "$NO_BELIEF" == "true" ]]; then
            cmd="$cmd --no-belief"
        fi
        if [[ "$record" == "true" ]]; then
            cmd="$cmd --record-baseline"
        elif [[ "$LIVE_EVAL" == "true" ]]; then
            # live-eval 只跟一般評估跑一次；基準那次只要 offline 數字。
            cmd="$cmd --live-eval --live-eval-games $LIVE_EVAL_GAMES"
            if [[ -n "$LIVE_EVAL_WORKERS" ]]; then
                cmd="$cmd --live-eval-workers $LIVE_EVAL_WORKERS"
            fi
        fi
        printf '%s' "$cmd"
    }

    echo "${#TRAIN_TARGETS[@]} model(s)"

    for target in "${TRAIN_TARGETS[@]}"; do
        out_dir="${target%%|*}"
        arch="${target##*|}"

        if [[ ! -f "$out_dir/ckpt-best.pt" ]]; then
            echo "  $(basename "$out_dir"): ⊘ (no ckpt-best.pt)"
            EVAL_FAILED+=("$out_dir (缺 ckpt-best.pt)")
            continue
        fi

        LABEL="${arch:+a$arch}"
        LABEL="${LABEL:-generalist}"
        echo -n "  $LABEL eval... "
        EVAL_LOG="$LOG_DIR/4c-eval${arch:+-a$arch}.log"
        set +e
        run_stage "$EVAL_LOG" "$(build_eval_cmd "$out_dir" "$arch" val false)"
        EVAL_RC=$?
        set -e
        if [[ $EVAL_RC -ne 0 ]]; then
            echo "✗"
            echo "[警告] 評估失敗: $out_dir — 末尾 20 行:" >&2
            tail -n 20 "$EVAL_LOG" >&2
            EVAL_FAILED+=("$out_dir")
            continue
        fi
        echo "✓"
        grep -E "Offline eval|Live eval vs" "$EVAL_LOG" || true

        echo -n "  $LABEL baseline... "
        BASE_LOG="$LOG_DIR/4c-baseline${arch:+-a$arch}.log"
        set +e
        run_stage "$BASE_LOG" "$(build_eval_cmd "$out_dir" "$arch" test true)"
        BASE_RC=$?
        set -e
        if [[ $BASE_RC -ne 0 ]]; then
            echo "✗"
            echo "[警告] 基準記錄失敗: $out_dir — 階段 5 的 IL 回歸檢查將被跳過" >&2
            tail -n 20 "$BASE_LOG" >&2
            EVAL_FAILED+=("$out_dir (baseline)")
            continue
        fi
        echo "✓"
        grep -E "Recorded IL baseline" "$BASE_LOG" || true
    done
fi

# ── 步驟 5: RL (R1 critic 修復 → R2 PPO + KL anchor) ────────────────────────
# 對每個訓練好的專家模型各跑一次，依序而非同時。RL_SPEC §10.1 決定用交替
# (alternating) 而非併行訓練：同時訓練兩個模型會讓雙方都變成非穩態，任何回歸
# 都變成耦合動力學問題，R2 的 go/no-go 也就無法歸因。
#
# 注意這裡「依序各跑一次」不等於 §10.1 的 league：沒有 cross-play、沒有過往
# 冠軍池、沒有 Elo。那是 R4，不在本階段範圍內。每個牌組各自對自己的 π_IL
# 做 self-play 並各自 gate，彼此獨立。
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
    # 預設只取「表現最好的 RL_TOP_N 副」，而不是每一副都跑：R1+R2 一副牌就要
    # 數小時，--n-archetypes all 之下全跑等於把預算平均灑在明顯較弱的牌組上。
    # 排名用 4c 寫進 data/il_baselines.json 的 held-out test 分數，並且只在
    # 真的訓練成功（有 ckpt-best.pt）的牌組裡挑。
    #
    # 牌組會跟著模型走：--il-ckpt 指向 checkpoints_a<N>/，--deck-archetype 也是
    # 同一個 N，ptcg_rl.train 再用它從 data/ 重建牌表，三者同源。
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
    echo "[5/5] RL (${RL_PHASE}): $RL_N deck(s) — $RL_TARGETS  [$RL_PICK]"

    for rl_arch in $RL_TARGETS; do
        RL_CKPT_DIR="${CHECKPOINT_DIR}_a${rl_arch}"
        RL_IL_CKPT="$RL_CKPT_DIR/ckpt-best.pt"
        RL_OUT_DIR="${RL_CKPT_DIR}_rl"

        stage_label "a${rl_arch}" true

        if [[ ! -f "$RL_IL_CKPT" ]]; then
            stage_done "a${rl_arch}" true "✗"
            echo "[錯誤] 找不到 $RL_IL_CKPT —— 階段 5 需要一個訓練好的專家模型" >&2
            RL_FAILED+=("a${rl_arch} (缺 ckpt-best.pt)")
            continue
        fi

        RL_CMD="uv run python -m ptcg_rl.train \
    --data-dir $DATA_DIR \
    --il-ckpt $RL_IL_CKPT \
    --out-dir $RL_OUT_DIR \
    --deck-archetype $rl_arch \
    --phase $RL_PHASE \
    --total-steps $RL_STEPS \
    --critic-steps $RL_CRITIC_STEPS \
    --n-workers $RL_WORKERS \
    --gate-games $RL_GATE_GAMES \
    --seed $SEED$RL_EXTRA"

        RL_LOG="$LOG_DIR/5-rl-a${rl_arch}.log"
        set +e
        run_stage "$RL_LOG" "$RL_CMD" true
        RL_RC=$?
        set -e
        if [[ $RL_RC -ne 0 ]]; then
            stage_done "a${rl_arch}" true "✗"
            fail_tail "$RL_LOG" "$RL_RC"
            # 一副牌失敗不該讓另一副的結果消失（它們互相獨立），但要記下來，
            # 而且整條管線最後要以非零退出。
            RL_FAILED+=("a${rl_arch}")
            continue
        fi
        stage_done "a${rl_arch}" true "✓"
        grep -E "R1 phase|R1 total|R1 gate|Gate:|R2 total|RL stage done" "$RL_LOG" | tail -n 6 || true
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
    echo "  RL model: $d/ckpt-rl-last.pt"
    echo "  RL report: $d/rl_report.json"
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
