#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — 完整训练管线 (Phase 1 → Phase 4 → Training)
#
# 用法:
#   ./scripts/run_pipeline.sh                        # 全部执行（下载 + 挖掘 + 建分片 + 训练）
#   ./scripts/run_pipeline.sh --skip-download          # 跳过下载（已有 raw/ 资料）
#   ./scripts/run_pipeline.sh --skip-download --no-train  # 只做到建分片，不训练
#
# 环境要求:
#   - uv (Python 3.11+)
#   - ~/.kaggle/kaggle.json (下载需要)
# =============================================================================

set -euo pipefail

# ── 预设参数 ────────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=""
NO_TRAIN=""
NO_EVAL=""
SKIP_RUST=""
# 预设安静模式：各阶段完整输出写入 logs/ 下的日志档，stdout 只印摘要。
# AI agent 跑这条管线时 stdout 太吵会塞爆 context window；要即时完整输出
# （例如在终端手动跑、想盯训练进度）加 --verbose。
VERBOSE=""
# 评估设定。离线评估（top-1/top-3）永远跑；live-eval 要真的开对局，慢得多，
# 所以预设关闭，用 --live-eval 打开。
LIVE_EVAL=""
# 对手牌组信念模型（辅助头）。预设关闭：旧的 shards 没有 belief 标签，
# 开了也只是加一个全被遮蔽的零项，白付 [B, V] 矩阵乘法的代价。
BELIEF=""
LIVE_EVAL_GAMES=200
LIVE_EVAL_WORKERS=""
# 相对路径一律以 python/ 为基准（管线全程在 python/ 下执行，
# 因为 ptcg_mine / ptcg_il 套件位于该目录，必须是 cwd 才 import 得到）。
RAW_DIR="raw"
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
N_DAYS=20
TARGET_EPISODES=10000
SEED=0
K_EXPERTS=10
G_MIN=50
JACCARD_THRESH=0.90
BATCH_SIZE=1024
EPOCHS=10
# 每个牌组训练一个专家模型（archetype specialist）。
# 六个 D_self 原型近乎互斥（pairwise multiset-Jaccard <= 0.17，没有任何一张卡同时
# 出现在全部六个里），单一通才模型必须同时拟合数个互不相干的策略，因此
# --archetype-self 的非平凡 top-1 提升是通才的 3~10 倍（arch 0: +2.7 -> +20.5；
# arch 2: +8.8 -> +49.4）。详见 CLAUDE.md「Key design decisions」。
# 每个原型输出到 ${CHECKPOINT_DIR}_a<id>/，各自带 decks.json / deck.csv 标签。
# 设为空字符串（或传 --generalist）则退回旧行为：只训练一个通才模型。
ARCHETYPES="0 2"
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
        --generalist)
            ARCHETYPES=""; shift
            ;;
        --no-eval)
            NO_EVAL="true"; shift
            ;;
        --belief)
            BELIEF="true"; shift
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
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "管线步骤: Rust engine → Phase 1-2(mine) → Phase 3(build-shards) → QA → Training → Eval"
            echo ""
            echo "路径说明: 相对路径以 python/ 为基准（预设 = python/raw, python/data,"
            echo "          python/checkpoints）。要指定别处请给绝对路径。"
            echo ""
            echo "选项:"
            echo "  --skip-download        跳过 Phase 1 下载（raw/ 已有资料时使用）"
            echo "  --skip-rust            跳过 Rust search engine 编译"
            echo "  --no-train             只做 mine + build-shards，不训练"
            echo "  --raw-dir DIR          raw 目录 (默认: python/raw)"
            echo "  --data-dir DIR         产出目录 (默认: python/data)"
            echo "  --checkpoint-dir DIR   checkpoint 目录 (默认: python/checkpoints)"
            echo "  --n-days N             下载的天数 (默认: 20)"
            echo "  --target-episodes N    总 episode 目标 (默认: 10000)"
            echo "  --seed N               随机种子 (默认: 0)"
            echo "  --batch-size N         训练 batch size (默认: 2048)"
            echo "  --epochs N             训练 epoch 数 (默认: 10)"
            echo "  --resume PATH          从指定 checkpoint 恢复训练"
            echo "  --archetypes \"0 2\"     为这些 archetype 各训练一个专家模型"
            echo "                         (默认: \"0 2\"，输出到 ${CHECKPOINT_DIR}_a<id>/)"
            echo "  --generalist           只训练单一通才模型 (旧行为，准确率明显较差)"
            echo "  --no-eval              训练后跳过评估 (步骤 5)"
            echo "  --belief               训练对手牌组信念头（archetype/deck/hidden/hand），"
            echo "                         并在 live eval 加上 search_planner_belief 对手。"
            echo "                         需要带 belief 标签的 shards（重新跑 build-shards）。"
            echo "  --live-eval            额外跑 live engine 对局评估 (慢: 每局都要真的打完)"
            echo "  --live-eval-games N    live-eval 每个对手的局数 (默认: 200)"
            echo "  --live-eval-workers N  live-eval 并行 worker 数 (默认: 由 CLI 决定)"
            echo "  --verbose, -v          各阶段完整输出直接印到终端 (默认: 写入日志档，"
            echo "                         stdout 只印摘要，避免 AI agent 的 context 被洗版)"
            echo "  --help, -h             显示此帮助"
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

# run_stage <logfile> <cmd-string>
# 安静模式：输出全部进 logfile；--verbose：tee 到终端与 logfile。
# 回传指令本身的 exit code（pipefail 之下 tee 管线也一样）。
run_stage() {
    local log="$1" cmd="$2"
    if [[ "$VERBOSE" == "true" ]]; then
        eval "$cmd" 2>&1 | tee "$log"
    else
        eval "$cmd" >"$log" 2>&1
    fi
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
echo -n "[2/5] Mine... "
MINE_CMD="uv run python -m ptcg_mine.mine \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --n-days $N_DAYS \
    --target-episodes $TARGET_EPISODES \
    --seed $SEED \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH \
    $SKIP_DOWNLOAD"

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
echo "✓"
# 抽出 mine 摘要的最后 6 行（避免整个 summary block 塞爆 context）
sed -n '/=== Corpus mining summary ===/,$p' "$MINE_LOG" | tail -n 6

# ── 步骤 3: Phase 3 特征化 ────────────────────────────────────────────────
echo -n "[3/5] Build shards... "
BS_CMD="uv run python -m ptcg_il.cli build-shards \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH"

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
echo "✓ ($SHARD_COUNT shards)"
grep "Shard build complete" "$BS_LOG" || true

# ── 步骤 4: 训练 ──────────────────────────────────────────────────────────
if [[ "$NO_TRAIN" == "true" ]]; then
    echo "═══ --no-train 已指定，跳过训练 ═══"
    echo "管线结束。产物在: $DATA_DIR/"
    exit 0
fi

echo -n "[4/5] Training... "

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
    if [[ "$BELIEF" == "true" ]]; then
        cmd="$cmd --belief"
    fi
    if [[ -n "${RESUME_CKPT:-}" ]]; then
        cmd="$cmd $RESUME_CKPT"
    fi
    printf '%s' "$cmd"
}

# 收集要训练的 (输出目录, archetype) 组合
TRAIN_TARGETS=()
if [[ -z "${ARCHETYPES// /}" ]]; then
    TRAIN_TARGETS+=("$CHECKPOINT_DIR|")
else
    for arch in $ARCHETYPES; do
        TRAIN_TARGETS+=("${CHECKPOINT_DIR}_a${arch}|${arch}")
    done
fi

echo "${#TRAIN_TARGETS[@]} model(s)"

TRAINED_DIRS=()
for target in "${TRAIN_TARGETS[@]}"; do
    out_dir="${target%%|*}"
    arch="${target##*|}"
    TRAIN_CMD="$(build_train_cmd "$out_dir" "$arch")"
    TRAIN_LOG="$LOG_DIR/4-train${arch:+-a$arch}.log"

    LABEL="${arch:+a$arch}"
    LABEL="${LABEL:-generalist}"
    echo -n "  $LABEL... "
    set +e
    run_stage "$TRAIN_LOG" "$TRAIN_CMD"
    TRAIN_RC=$?
    set -e
    if [[ $TRAIN_RC -ne 0 ]]; then
        echo "✗"
        fail_tail "$TRAIN_LOG" "$TRAIN_RC"
        exit $TRAIN_RC
    fi
    echo "✓"
    # 训练摘要：最后几次 eval + 完成行
    grep -E "eval step|Training complete" "$TRAIN_LOG" | tail -n 3 || true
    TRAINED_DIRS+=("$out_dir")
done

# ── 步骤 5: 评估 ──────────────────────────────────────────────────────────
EVAL_FAILED=()
if [[ "$NO_EVAL" == "true" ]]; then
    echo "═══ --no-eval — 跳过评估 ═══"
else
    echo -n "[5/5] Eval... "

    build_eval_cmd() {
        local out_dir="$1" arch="$2"
        local cmd="uv run python -m ptcg_il.cli train \
    --eval-only \
    --data-dir $DATA_DIR \
    --out-dir $out_dir \
    --resume $out_dir/ckpt-best.pt"
        if [[ -n "$arch" ]]; then
            cmd="$cmd --archetype-self $arch"
        fi
        if [[ "$BELIEF" == "true" ]]; then
            cmd="$cmd --belief"
        fi
        if [[ "$LIVE_EVAL" == "true" ]]; then
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

        EVAL_CMD="$(build_eval_cmd "$out_dir" "$arch")"
        EVAL_LOG="$LOG_DIR/5-eval${arch:+-a$arch}.log"
        LABEL="${arch:+a$arch}"
        LABEL="${LABEL:-generalist}"
        echo -n "  $LABEL... "
        set +e
        run_stage "$EVAL_LOG" "$EVAL_CMD"
        EVAL_RC=$?
        set -e
        if [[ $EVAL_RC -ne 0 ]]; then
            echo "✗"
            echo "[警告] 评估失败: $out_dir — 末尾 20 行:" >&2
            tail -n 20 "$EVAL_LOG" >&2
            EVAL_FAILED+=("$out_dir")
            continue
        fi
        echo "✓"
        grep -E "Offline eval|Live eval vs" "$EVAL_LOG" || true
    done
fi

echo ""
echo "=== done ==="
for d in "${TRAINED_DIRS[@]}"; do
    if [[ -f "$d/ckpt-best.pt" ]]; then
        echo "  model: $d/ckpt-best.pt"
    else
        echo "  model: $d/ckpt-best.pt  [不存在]"
    fi
done
echo "  data:  $DATA_DIR/"
echo "  logs:  $LOG_DIR/"

if [[ "${#EVAL_FAILED[@]}" -gt 0 ]]; then
    echo ""
    echo "[警告] 评估失败/跳过:" >&2
    for f in "${EVAL_FAILED[@]}"; do
        echo "         - $f" >&2
    done
fi

# 评估失败 ≠ 整条管线失败，但要让呼叫者看得见
if [[ "${#EVAL_FAILED[@]}" -gt 0 ]]; then
    exit 1
fi
