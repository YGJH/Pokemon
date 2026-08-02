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
#
# 會自動編譯 Rust MCTS library (libptcg_search.so) 並打包進 submission。
# 如果 cargo 找不到，會跳過並在 submission 中使用 greedy fallback。
#
# --no-mcts: 每個決策只跑一次 policy forward pass，不做樹搜尋。不編譯也不打包
# libptcg_search.so，bundle 裡也不會有 search_infer.py / belief_posterior.py /
# archetypes.json。輸出檔名改成 submission-greedy.tar.gz，才不會跟 MCTS 版本
# 互相覆蓋 —— 兩個 bundle 長得一模一樣，覆蓋掉就分不出上傳的是哪一個。

set -euo pipefail

NO_MCTS=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --no-mcts) NO_MCTS=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

ARCH="${POSITIONAL[0]:-a0}"
EXPLICIT_CKPT="${POSITIONAL[1]:-}"

IL_DIR="python/checkpoints_${ARCH}"
MCTS_DIR="python/checkpoints_${ARCH}_mcts"
ELO_FILE="${MCTS_DIR}/elo_ratings.json"
echo "$ELO_FILE"
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

# ── Build Rust MCTS library ─────────────────────────────────────────────
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

if [[ "$NO_MCTS" == 1 ]]; then
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        --no-mcts \
        --out submission-greedy.tar.gz
else
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        --out submission.tar.gz
fi
