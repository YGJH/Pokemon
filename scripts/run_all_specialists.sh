#!/usr/bin/env bash
#
# Phase 2 — Specialist MCTS Branching.
#
# 六個 branch，每個 branch 一副牌、一組獨立權重，全部從同一個通才模型
# (foundation model) warm-start。一次只訓練一個 branch；輪流跑，每輪每副牌
# ROUND_GAMES 場自我對弈。
#
#   Phase 1（跑一次，本腳本的前提）：
#       ./scripts/run_pipeline.sh --skip-download --generalist
#     產出 python/checkpoints_generalist/ckpt-best.pt
#
#   Phase 2（本腳本，跑到手動中斷）：
#       ./scripts/run_all_specialists.sh
#
# 關鍵差別在 seat 1。以前 θ 同時操作兩個座位，而 seat 1 拿的是「抽樣到的」
# 對手牌組 —— θ 是某一副牌的專家，等於拿它沒見過的 60 張卡在打，勝率
# 0.74 量到的是對手不會打，不是 θ 強。現在每副牌由「擁有那副牌的 expert」
# 操作（checkpoints_a<M>_mcts 最新的 champion），沒有 expert 的牌組退回通才。
# 對手牌池仍然是全部 179 副：Kaggle 的對手本來就是分布外的，這裡只換「誰來
# 操作」，沒有縮小抽樣範圍。
#
# expert 是跨輪更新的：branch N 在第 r+1 輪面對的，是第 r 輪練強過的對手。
# 每個 branch 開跑前才重算 --opp-expert 路徑，所以同一輪裡排在後面的 branch
# 也會看到前面剛更新的對手。
#
# 中斷安全：champion 只在通過 gate 時才寫出，mcts_train 自己會從 out-dir 裡
# ELO 最高的 champion 續跑，所以 Ctrl-C 不會弄丟已經通過的成果，重跑本腳本
# 就是接著練。
#
# 環境變數可覆寫的參數列在下面「可調參數」區。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY_DIR="$PROJECT_DIR/python"

# ── 可調參數 ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
GENERALIST_CKPT="${GENERALIST_CKPT:-${CHECKPOINT_DIR}_generalist/ckpt-best.pt}"

N_DECKS="${N_DECKS:-6}"          # 幾副牌開 branch
# 有多少比例的對局，對手牌組直接從「其他 branch 的牌」裡抽（由該牌的 expert
# 操作）。0 的話 expert 只有在自己那副牌剛好被抽中時才會上場 —— 在 201 副的
# 牌池裡是 2%，看起來像 round-robin，其實一輪 200 場只碰到 4 場。
# 剩下的比例仍然抽全牌池：實戰對手本來就是分布外的，只練那 5 副會過擬合一個
# 只存在於這個迴圈裡的 metagame。
OPP_EXPERT_SHARE="${OPP_EXPERT_SHARE:-0.5}"
ROUND_GAMES="${ROUND_GAMES:-1000}"   # 每個 branch 每輪的自我對弈場數
MAX_ROUNDS="${MAX_ROUNDS:-0}"    # 0 = 跑到手動中斷
DECKS="${DECKS:-}"               # 非空則直接用這串 id，跳過推導

# MCTS 搜尋 / rollout。取自 scripts/run.sh 目前實際在跑的那組值。
MCTS_ITERATIONS="${MCTS_ITERATIONS:-16}"
MCTS_C_PUCT="${MCTS_C_PUCT:-2.0}"
MCTS_K_DET="${MCTS_K_DET:-4}"
MCTS_LEAF_BATCH="${MCTS_LEAF_BATCH:-128}"
MCTS_RHO="${MCTS_RHO:-1.0}"
MCTS_N_ENGINES="${MCTS_N_ENGINES:-20}"
MCTS_WORKERS="${MCTS_WORKERS:-16}"
MCTS_FORWARD_BATCH="${MCTS_FORWARD_BATCH:-1024}"

MCTS_BUFFER_CAPACITY="${MCTS_BUFFER_CAPACITY:-10000}"
MCTS_MIN_BUFFER="${MCTS_MIN_BUFFER:-1000}"
MCTS_GAMES_PER_ITER="${MCTS_GAMES_PER_ITER:-200}"
MCTS_TRAIN_STEPS_PER_ITER="${MCTS_TRAIN_STEPS_PER_ITER:-200}"
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-128}"
MCTS_LR="${MCTS_LR:-0.00002}"
# constant，不是 cosine。--total-games 現在是「一輪」的預算，cosine 會把
# 整段 warmup→衰減塞進每一輪，等於每輪做一次 warm restart —— 那是換排程，
# 不是換預算。之前 --total-games 5000000 的 cosine 實質上就是常數。
MCTS_LR_SCHEDULE="${MCTS_LR_SCHEDULE:-constant}"
MCTS_C_VALUE="${MCTS_C_VALUE:-0.5}"
MCTS_C_PI="${MCTS_C_PI:-1.0}"
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
MCTS_GRAD_CLIP="${MCTS_GRAD_CLIP:-30}"

MCTS_EVAL_GAMES="${MCTS_EVAL_GAMES:-100}"
MCTS_EVAL_EVERY_GAMES="${MCTS_EVAL_EVERY_GAMES:-100}"
MCTS_GATE_SCORE="${MCTS_GATE_SCORE:-0.70}"
MCTS_MAX_CHAMPIONS="${MCTS_MAX_CHAMPIONS:-8}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pokemon-tcg-mcts}"
WANDB_ENTITY="${WANDB_ENTITY:-poken}"

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/branch}"

# ── 參數解析 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --decks)        DECKS="$2"; shift 2 ;;
        --n-decks)      N_DECKS="$2"; shift 2 ;;
        --round-games)  ROUND_GAMES="$2"; shift 2 ;;
        --expert-share) OPP_EXPERT_SHARE="$2"; shift 2 ;;
        --rounds)       MAX_ROUNDS="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --no-wandb)     WANDB=0; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            echo ""
            echo "選項:"
            echo "  --decks \"1 17 25\"   直接指定 archetype id，跳過自動推導"
            echo "  --n-decks N          開幾個 branch (預設: 6)"
            echo "  --round-games N      每個 branch 每輪的自我對弈場數 (預設: 10000)"
            echo "  --expert-share F     多少比例的對局對上其他 branch 的 expert (預設: 0.5)"
            echo "  --rounds N           跑幾輪後結束 (預設: 0 = 跑到手動中斷)"
            echo "  --device DEV         cuda / cpu (預設: cuda)"
            echo "  --no-wandb           關閉 W&B"
            echo "  --dry-run            只印出每個 branch 的指令，不執行"
            exit 0
            ;;
        *) echo "[錯誤] 未知選項: $1" >&2; exit 1 ;;
    esac
done
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"
cd "$PY_DIR"

# ── 前提：Phase 1 的通才模型 ─────────────────────────────────────────────
if [[ ! -f "$GENERALIST_CKPT" ]]; then
    echo "[錯誤] 找不到 $PY_DIR/$GENERALIST_CKPT" >&2
    echo "       Phase 1 還沒跑。先執行：" >&2
    echo "         ./scripts/run_pipeline.sh --skip-download --generalist" >&2
    exit 1
fi

# ── 編譯 Rust ────────────────────────────────────────────────────────────
if command -v cargo &>/dev/null && [[ -f "ptcg_search/Cargo.toml" ]]; then
    echo -n "  cargo build --release ... "
    if (cd ptcg_search && cargo build --release >"$LOG_DIR/cargo.log" 2>&1); then
        echo "✓"
    else
        echo "✗ (見 $LOG_DIR/cargo.log)"
        exit 1
    fi
fi

# ── 要開 branch 的牌組 ───────────────────────────────────────────────────
# archetype id 是分群索引，重跑 mine 會重新編號 —— 從 artifacts 推導，不寫死。
if [[ -z "${DECKS// /}" ]]; then
    echo -n "  archetypes: "
    set +e
    RAW=$(uv run python -m ptcg_il.cli archetypes \
              --data-dir "$DATA_DIR" --top "$N_DECKS" \
              2>"$LOG_DIR/archetypes.log")
    ARCH_RC=$?
    set -e
    if [[ $ARCH_RC -ne 0 ]]; then
        echo "✗"
        cat "$LOG_DIR/archetypes.log" >&2
        echo "[提示] uv run python -m ptcg_il.cli archetypes --data-dir $DATA_DIR --describe" >&2
        exit 1
    fi
    # cli.py 的 RichHandler 寫的是 stdout，不是 stderr —— 「只有 5 副牌有足夠
    # held-out 資料」這類 WARNING 會跟 id 混在同一條 pipe 裡。所以只取「整行
    # 都是空白分隔的整數」的最後一行，而不是整包 stdout；否則 DECKS 會混進
    # 一堆單字，for 迴圈就會拿 "WARNING" 當 archetype id 去跑。
    DECKS=$(printf '%s\n' "$RAW" | grep -E '^[0-9]+( +[0-9]+)*$' | tail -n 1)
    echo "${DECKS:-✗}"
    printf '%s\n' "$RAW" > "$LOG_DIR/archetypes.stdout"
fi
if [[ -z "${DECKS// /}" ]]; then
    echo "[錯誤] 推導不出任何 archetype (完整輸出見 $LOG_DIR/archetypes.stdout)" >&2
    exit 1
fi
# 明確擋掉非數字的 id：archetype id 一路傳到 --deck-archetype 和 --opp-expert，
# 混進一個單字只會在幾分鐘後才炸在 argparse 或 build_deck_metadata 裡。
for arch in $DECKS; do
    if ! [[ "$arch" =~ ^[0-9]+$ ]]; then
        echo "[錯誤] '$arch' 不是 archetype id (DECKS='$DECKS')" >&2
        exit 1
    fi
done

N_BRANCH=$(echo "$DECKS" | wc -w)

# ── 某副牌目前的 expert checkpoint ───────────────────────────────────────
# 只認 champion：champion 是通過 gate 才寫出來的，ckpt-mcts-last.pt 不是成果。
# 檔名是零填充的 champion-%06d，所以字典序等於場數序。
# 沒有 champion 就印空字串 —— 呼叫端會讓它退回通才。
# 一定要 return 0：第一輪每個 branch 都還沒有 champion，ls 會以 2 結束，而
# `x="$(expert_ckpt_for ...)"` 的結束碼就是命令替換的結束碼 —— 在 set -e 底下
# 整個腳本會在「還沒有對手可用」這個完全正常的狀態下直接死掉。
expert_ckpt_for() {
    # 兩行，不能併成 `local arch="$1" dir="..${arch}.."`：bash 會先把整條
    # local 命令的所有 word 展開完才做賦值，所以那個 ${arch} 取到的是呼叫端
    # 迴圈裡的 arch（正在訓練的那副牌），不是 $1。合寫的結果是每個 branch 把
    # 自己的 champion 當成全部對手的 expert —— 完全沒有錯誤訊息。
    local arch="$1"
    local dir="${CHECKPOINT_DIR}_a${arch}_mcts"
    [[ -d "$dir" ]] || return 0
    local found
    found=$(printf '%s\n' "$dir"/ckpt-mcts-champion-*.pt \
                | grep -v '\*' | sort | tail -n 1) || true
    [[ -n "$found" && -f "$found" ]] && printf '%s' "$found"
    return 0
}

echo "═══ Phase 2: Specialist MCTS Branching ═══"
echo "  foundation : $GENERALIST_CKPT"
echo "  branches   : $N_BRANCH — $DECKS"
echo "  round      : $ROUND_GAMES games/branch"
echo "  rival share: $OPP_EXPERT_SHARE"
echo "  rounds     : $([ "$MAX_ROUNDS" -eq 0 ] && echo '∞ (Ctrl-C 結束)' || echo "$MAX_ROUNDS")"
echo "  logs       : $LOG_DIR"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

round=0
while [[ "$MAX_ROUNDS" -eq 0 ]] || [[ "$round" -lt "$MAX_ROUNDS" ]]; do
    round=$((round + 1))
    echo "━━━ round $round ━━━"

    for arch in $DECKS; do
        OUT_DIR="${CHECKPOINT_DIR}_a${arch}_mcts"

        # 對手席位的 pilot。每個 branch 開跑前才重算，所以同一輪裡排在後面的
        # branch 會看到前面剛寫出的 champion。
        OPP_FLAGS=()
        PILOTED=()
        for other in $DECKS; do
            [[ "$other" == "$arch" ]] && continue
            other_ckpt="$(expert_ckpt_for "$other")"
            if [[ -n "$other_ckpt" ]]; then
                OPP_FLAGS+=(--opp-expert "${other}=${other_ckpt}")
                PILOTED+=("a${other}")
            fi
        done

        echo "  [round $round] a${arch}  seat-1 experts: ${PILOTED[*]:-none (全部退回通才)}"

        CMD=(uv run python -m ptcg_rl.mcts_train
            --data-dir "$DATA_DIR"
            --il-ckpt "$GENERALIST_CKPT"
            --init-ckpt "$GENERALIST_CKPT"
            --opp-default-ckpt "$GENERALIST_CKPT"
            --opp-expert-share "$OPP_EXPERT_SHARE"
            "${OPP_FLAGS[@]}"
            --out-dir "$OUT_DIR"
            --deck-archetype "$arch"
            --iterations "$MCTS_ITERATIONS"
            --c-puct "$MCTS_C_PUCT"
            --k-determinizations "$MCTS_K_DET"
            --leaf-batch "$MCTS_LEAF_BATCH"
            --rho "$MCTS_RHO"
            --mcts-distill
            --n-engines "$MCTS_N_ENGINES"
            --n-workers "$MCTS_WORKERS"
            --forward-batch "$MCTS_FORWARD_BATCH"
            --buffer-capacity "$MCTS_BUFFER_CAPACITY"
            --min-buffer "$MCTS_MIN_BUFFER"
            --total-games "$ROUND_GAMES"
            --games-per-iter "$MCTS_GAMES_PER_ITER"
            --train-steps-per-iter "$MCTS_TRAIN_STEPS_PER_ITER"
            --batch-size "$MCTS_BATCH_SIZE"
            --lr "$MCTS_LR"
            --lr-schedule "$MCTS_LR_SCHEDULE"
            --c-value "$MCTS_C_VALUE"
            --c-pi "$MCTS_C_PI"
            --grad-clip "$MCTS_GRAD_CLIP"
            --all-archetypes
            --eval-games "$MCTS_EVAL_GAMES"
            --eval-every-games "$MCTS_EVAL_EVERY_GAMES"
            --gate-score "$MCTS_GATE_SCORE"
            --max-champions "$MCTS_MAX_CHAMPIONS"
            --seed "$SEED"
            --device "$DEVICE"
        )
        if [[ "$WANDB" == "1" ]]; then
            CMD+=(--wandb --wandb-project "$WANDB_PROJECT"
                  --wandb-entity "$WANDB_ENTITY"
                  --wandb-name "${WANDB_PROJECT}-a${arch}")
        else
            CMD+=(--no-wandb)
        fi

        if [[ "$DRY_RUN" == "1" ]]; then
            printf '    %q ' "${CMD[@]}"; echo ""
            continue
        fi

        BRANCH_LOG="$LOG_DIR/r${round}-a${arch}.log"
        # 一個 branch 掛掉不該讓整輪停下來：其他牌組的成果是獨立的，而且
        # champion 已經寫在各自的 out-dir 裡。記下來，這一輪結束再報。
        set +e
        "${CMD[@]}" 2>&1 | tee "$BRANCH_LOG"
        RC=${PIPESTATUS[0]}
        set -e
        if [[ $RC -ne 0 ]]; then
            echo "  [round $round] a${arch} ✗ (exit $RC) — 見 $BRANCH_LOG" >&2
        else
            latest="$(expert_ckpt_for "$arch")"
            echo "  [round $round] a${arch} ✓  champion: ${latest:-[本輪沒有通過 gate]}"
        fi
    done
done

echo "=== done ($round rounds) ==="
for arch in $DECKS; do
    latest="$(expert_ckpt_for "$arch")"
    echo "  a${arch}: ${latest:-[沒有 champion 通過 gate]}"
done
