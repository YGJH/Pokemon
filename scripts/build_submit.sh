#!/bin/bash
# 打包 Kaggle 提交檔。
#
# 預設自動挑選 ELO 最高的模型（從 elo_ratings.json 讀取），找不到才退回
# ckpt-best.pt。也可以手動指定 checkpoint 路徑。
#
# checkpoint 一定要是「用現在這份 python/data 訓練出來的」那一個。
# archetype id 只是分群索引，重跑 mine 就會重新編號，所以 checkpoints_a<N>/
# 的 N 不代表任何固定的牌組 —— 舊語料留下來的目錄看起來完全正常，卻是對著
# 另一份 vocab 訓練的。build_submission.py 現在會比對 checkpoint 裡釘住的
# vocab/archetypes sha1，對不上就中止，而不是打包出一個「跑得完但幾乎全輸」
# 的 bundle。
#
# 用法:
#   ./scripts/build_submit.sh                         # a0, 自動選最高 ELO
#   ./scripts/build_submit.sh a1                      # a1, 自動選最高 ELO
#   ./scripts/build_submit.sh a0 path/to/ckpt.pt      # 手動指定 checkpoint
#   ./scripts/build_submit.sh a0 --no-mcts            # 純 policy，不含任何搜尋
#   ./scripts/build_submit.sh --ensemble "..." --mcts  # ensemble+MCTS+對手牌組先驗
#   ./scripts/build_submit.sh --ensemble "python/checkpoints_a1_s*/ckpt-best.pt" \
#       --ensemble-top 7                              # 10 個成員裡挑最好的 7 個（greedy）
#   ./scripts/build_submit.sh --ensemble "python/checkpoints_a1_s*/ckpt-best.pt" \
#       --ensemble-top 2 --mcts                       # ensemble + MCTS + 對手牌組先驗推斷
#
# --ensemble-top N: Kaggle 限制一次能提交幾個 checkpoint，所以 10 個成員要砍到
# N 個。挑法是讀 data/il_baselines.json 裡記錄的 greedy forward selection 順序
# —— 不是 glob 順序，也不是「單模型分數最高的 N 個」：ensemble 靠的是成員之間
# 不相關，所以單獨最好的 7 個不等於合起來最好的 7 個。
# 那份順序要先跑這行才會存在（在 val 上挑，再用 test 驗證選出來的子集）:
#   cd python && uv run python -m ptcg_il.cli train --eval-only \
#       --data-dir data --archetype-self 1 --eval-split val --ensemble-select 7 \
#       --ckpt checkpoints_a1_s0/ckpt-best.pt --ckpt ... （10 個都要帶）
# 沒有記錄就直接中止，不會安靜地拿前 N 個 —— 那會打包出一個「看起來有挑過」
# 但其實沒有的 submission，而且後面沒有任何一步會拆穿它。
#
# 會自動編譯 Rust MCTS library (libptcg_search.so) 並打包進 submission。
# 如果 cargo 找不到，會跳過並在 submission 中使用 greedy fallback。
#
# --no-mcts: 每個決策只跑一次 policy forward pass，不做樹搜尋。不編譯也不打包
# libptcg_search.so，bundle 裡也不會有 search_infer.py / belief_posterior.py /
# archetypes.json。輸出檔名改成 submission-greedy.tar.gz，才不會跟 MCTS 版本
# 互相覆蓋 —— 兩個 bundle 長得一模一樣，覆蓋掉就分不出上傳的是哪一個。

set -euo pipefail

ENSEMBLE=0
ENSEMBLE_PATHS=()
ENSEMBLE_TOP=""
ENSEMBLE_MCTS=0
FORCE=0
DATA_DIR="python/data"
NO_MCTS=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-mcts) NO_MCTS=1; shift ;;
        --force) FORCE=1; shift ;;
        --mcts) ENSEMBLE_MCTS=1; shift ;;
        --ensemble-top) ENSEMBLE_TOP="${2:?--ensemble-top needs a number}"; shift 2 ;;
        --ensemble)
            ENSEMBLE=1
            shift
            # Collect all remaining args as ensemble paths
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENSEMBLE_PATHS+=("$1")
                shift
            done
            ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# Expand globs in ensemble paths
if [[ "$ENSEMBLE" == 1 ]]; then
    EXPANDED_PATHS=()
    for p in "${ENSEMBLE_PATHS[@]}"; do
        for f in $p; do
            if [[ -f "$f" ]]; then
                EXPANDED_PATHS+=("$f")
            fi
        done
    done
    if [[ ${#EXPANDED_PATHS[@]} -eq 0 ]]; then
        echo "ERROR: --ensemble specified but no files matched" >&2
        exit 1
    fi
    ENSEMBLE_PATHS=("${EXPANDED_PATHS[@]}")

    # ── --ensemble-top: keep the best N of the matched members ──────────────
    # Kaggle caps how many checkpoints a submission may carry, so a 10-seed
    # ensemble has to be cut down.  The subset comes from the greedy ordering
    # recorded in il_baselines.json, *not* from the glob order and not from
    # per-member scores: an ensemble gains from decorrelated members, so the
    # best N individually is not the best N together.
    #
    # No fallback on purpose.  Quietly taking the first N of a glob would
    # produce a submission that looks selected and is not, and nothing
    # downstream ever contradicts it.
    if [[ -n "$ENSEMBLE_TOP" ]]; then
        if [[ "$ENSEMBLE_TOP" -ge "${#ENSEMBLE_PATHS[@]}" ]]; then
            echo "--ensemble-top $ENSEMBLE_TOP >= ${#ENSEMBLE_PATHS[@]} matched members — keeping all"
        else
            # ptcg_il lives under python/, so the selector runs from there and
            # the checkpoint paths have to be absolute to survive the cd.
            CKPT_ARGS_SEL=()
            for p in "${ENSEMBLE_PATHS[@]}"; do
                CKPT_ARGS_SEL+=(--ckpt "$(realpath "$p")")
            done
            # Resolved out here: inside the $( cd python && … ) subshell a
            # relative python/data would be looked up from python/.
            DATA_DIR_ABS="$(realpath "$DATA_DIR")"
            echo "Selecting top $ENSEMBLE_TOP of ${#ENSEMBLE_PATHS[@]} members..."
            SELECTED=$(cd python && uv run python -m ptcg_il.ensemble_select \
                --data-dir "$DATA_DIR_ABS" \
                --top "$ENSEMBLE_TOP" "${CKPT_ARGS_SEL[@]}") || {
                echo "ERROR: could not resolve --ensemble-top $ENSEMBLE_TOP" >&2
                exit 1
            }
            mapfile -t ENSEMBLE_PATHS <<< "$SELECTED"
        fi
    fi

    echo "Ensemble: ${#ENSEMBLE_PATHS[@]} members"
    for p in "${ENSEMBLE_PATHS[@]}"; do
        echo "  $p"
    done
fi

# --ensemble 自己帶了成員清單，下面整段單一 checkpoint 的挑選邏輯在那個模式
# 下算出來的 $CKPT 從頭到尾沒人用。留著只會做兩件壞事：印一段講 a0/ELO 的訊息，
# 讓人以為打包的是那個 checkpoint；以及用「$CKPT 不存在」中止一次跟它無關的打包。
if [[ "$ENSEMBLE" == 0 ]]; then

ARCH="${POSITIONAL[0]:-a0}"
EXPLICIT_CKPT="${POSITIONAL[1]:-}"

IL_DIR="python/checkpoints_${ARCH}"
MCTS_DIR="python/checkpoints_${ARCH}_mcts"
ELO_FILE="${MCTS_DIR}/elo_ratings.json"
if [[ -n "$EXPLICIT_CKPT" ]]; then
    # ── 手動指定 checkpoint ─────────────────────────────────────────────
    CKPT="$EXPLICIT_CKPT"
    echo "使用指定的 checkpoint: $CKPT"
elif [[ -f "$ELO_FILE" ]]; then
    # ── 從 elo_ratings.json 挑選 ELO 最高的模型 ─────────────────────────
    # Python 把結果寫到 stdout（只有路徑或 FALLBACK），其他資訊寫到 stderr
    CKPT=$(uv run python -c "
import json, sys
from pathlib import Path

elo_path = Path('$ELO_FILE')
data = json.loads(elo_path.read_text())
ratings = data.get('ratings', {})
fixed = set(data.get('fixed', []))
candidates = [(r, n) for n, r in ratings.items() if n not in fixed]
if not candidates:
    print('FALLBACK')
    sys.exit(0)

candidates.sort(reverse=True)
best_elo, best_name = candidates[0]
print(f'ELO 最高: {best_name} ({best_elo:.0f})', file=sys.stderr)

# 找出實際檔案路徑。
# elo_ratings.json 的 key 混用兩種寫法（'ckpt-best' 跟 'ckpt-best.pt'），
# 無條件補 .pt 會組出 'ckpt-step-0002000.pt.pt' —— 檔案永遠找不到，於是每次
# 都安靜退回 ckpt-best.pt，看起來就像「ELO 選擇失敗」而不是路徑組錯。
stem = best_name[:-3] if best_name.endswith('.pt') else best_name
il_dir = Path('$IL_DIR')
mcts_dir = Path('$MCTS_DIR')
for d in (mcts_dir, il_dir):
    p = d / f'{stem}.pt'
    if p.exists():
        print(str(p))
        sys.exit(0)

# 找不到檔案，列出搜尋過的目錄
print(f'找不到 ELO 最高模型 {best_name} 的檔案', file=sys.stderr)
print(f'  搜尋過: {mcts_dir}, {il_dir}', file=sys.stderr)
print('FALLBACK')
")

    if [[ "$CKPT" == "FALLBACK" ]] || [[ -z "$CKPT" ]]; then
        CKPT="${IL_DIR}/ckpt-best.pt"
        echo "自動選擇失敗，退回 $CKPT"
    else
        echo "自動選擇 ELO 最高模型: $CKPT"
    fi
else
    # ── 沒有 ELO 檔案，退回 ckpt-best ───────────────────────────────────
    CKPT="${IL_DIR}/ckpt-best.pt"
    echo "沒有 ELO 資料 ($ELO_FILE 不存在)，退回 $CKPT"
fi

if [[ ! -f "$CKPT" ]]; then
    echo "找不到 $CKPT" >&2
    echo "現有的 checkpoint:" >&2
    ls -d python/checkpoints* 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

fi  # ENSEMBLE == 0

# ── Build Rust MCTS library ─────────────────────────────────────────────
# Ensemble without --mcts keeps backward compat: MCTS was always off before
# ensemble support was added, so default to greedy unless explicitly requested.
if [[ "$ENSEMBLE" == 1 && "$ENSEMBLE_MCTS" == 0 ]]; then
    NO_MCTS=1
fi

RUST_DIR="python/ptcg_search"
if [[ "$NO_MCTS" == 1 ]]; then
    echo "--no-mcts: 跳過 Rust build，打包純 policy submission"
elif command -v cargo &>/dev/null && [[ -f "$RUST_DIR/Cargo.toml" ]]; then
    echo "Building libptcg_search.so..."
    (cd "$RUST_DIR" && cargo build --release 2>&1) || echo "WARNING: cargo build failed — submission will use greedy fallback"
else
    echo "WARNING: cargo not found — skipping Rust build, MCTS will fall back to greedy policy"
fi
# ─────────────────────────────────────────────────────────────────────────

if [[ "$ENSEMBLE" == 1 ]]; then
    CKPT_ARGS=()
    for p in "${ENSEMBLE_PATHS[@]}"; do
        CKPT_ARGS+=(--ckpt "$p")
    done
    MCTS_FLAG=()
    if [[ "$NO_MCTS" == 1 ]]; then
        MCTS_FLAG=(--no-mcts)
    fi
    FORCE_FLAG=()
    if [[ "$FORCE" == 1 ]]; then
        FORCE_FLAG=(--force)
    fi
    PREFIX="submission"
    if [[ "$NO_MCTS" == 1 ]]; then
        PREFIX="submission-greedy"
    fi
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        "${CKPT_ARGS[@]}" \
        "${MCTS_FLAG[@]}" \
        "${FORCE_FLAG[@]}" \
        --out "${PREFIX}-ens${#ENSEMBLE_PATHS[@]}.tar.gz"
elif [[ "$NO_MCTS" == 1 ]]; then
    FORCE_FLAG=()
    if [[ "$FORCE" == 1 ]]; then
        FORCE_FLAG=(--force)
    fi
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        --no-mcts \
        "${FORCE_FLAG[@]}" \
        --out submission-greedy.tar.gz
else
    FORCE_FLAG=()
    if [[ "$FORCE" == 1 ]]; then
        FORCE_FLAG=(--force)
    fi
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        "${FORCE_FLAG[@]}" \
        --out submission.tar.gz
fi
