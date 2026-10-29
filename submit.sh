#!/usr/bin/env bash
# Deploy + submit for the CUHK CSE GPU cluster (login node linux1).
# Run this from your Mac with the CSE VPN connected.
#
#   ./submit.sh                                  # train.py, defaults from config.py
#   ./submit.sh --epochs 10 --lr 0.02            # flags are forwarded to train.py
#   ./submit.sh --job eval --split validation    # evaluate.py instead
#   ./submit.sh --follow --epochs 3              # submit, wait, print the output
#   ./submit.sh --fetch-only                     # pull checkpoints/scores back only
#   ./submit.sh --dry-run --epochs 3             # show commands, touch nothing
#
# Consumed here (everything else is forwarded to the remote script):
#   -j, --job train|eval  remote entry point (default: train)
#   -f, --follow          wait for the job, then tail its slurm output
#   --fetch               copy checkpoints/ and scores*.csv back after submitting
#   --fetch-only          only copy them back; submit nothing
#   --no-sync             skip the rsync, submit what is already on the cluster
#   --dry-run             print the rsync/ssh/sbatch commands and exit
#   -h, --help            this text
#
# Env overrides:
#   REMOTE       ssh target              (default: cyho2@linux1)
#   REMOTE_DIR   project dir on cluster  (default: aist4010_project)
#   SSH_KEY      private key             (default: ./.ssh/id_ed25519)
#   FEATURES_DIR feature folder to use   (default: the one under ./features)
#
# Only features/ + the source files are synced (~10 MB); the dataset and the
# featurisation cache stay on this machine, which is the point of
# prepare_features.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE:-cyho2@linux1}"
REMOTE_DIR="${REMOTE_DIR:-aist4010_project}"
JOB="train"
FOLLOW=0
FETCH=0
FETCH_ONLY=0
SYNC=1
DRY_RUN=0
JOB_ARGS=()

usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -j|--job)     JOB="${2:-}"; shift 2 ;;
        -f|--follow)  FOLLOW=1; shift ;;
        --fetch)      FETCH=1; shift ;;
        --fetch-only) FETCH=1; FETCH_ONLY=1; shift ;;
        --no-sync)    SYNC=0; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            JOB_ARGS+=("$1"); shift ;;
    esac
done

case "$JOB" in
    train) TARGET="train.py" ;;
    eval|evaluate) TARGET="evaluate.py" ;;
    *) echo "unknown --job '$JOB' (use train or eval)" >&2; exit 1 ;;
esac

# --- 1. find a key ---------------------------------------------------------
KEY="${SSH_KEY:-}"
if [ -z "$KEY" ]; then
    for candidate in "$HERE/.ssh/id_ed25519" "$HOME/.ssh/id_ed25519"; do
        if [ -f "$candidate" ]; then KEY="$candidate"; break; fi
    done
fi
if [ -z "$KEY" ] || [ ! -f "$KEY" ]; then
    echo "no ssh key: expected $HERE/.ssh/id_ed25519 (or set SSH_KEY=...)" >&2
    exit 1
fi
if ! grep -q "PRIVATE KEY" "$KEY" 2>/dev/null; then
    echo "warning: $KEY does not look like an OpenSSH private key" >&2
fi

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

# --- 2. is the cluster reachable? -----------------------------------------
if [ "$DRY_RUN" -eq 0 ]; then
    if ! "${SSH_CMD[@]}" "$REMOTE" 'true' 2>/dev/null; then
        echo "cannot ssh to $REMOTE -- connect to the CSE VPN first (or set REMOTE=...)" >&2
        exit 1
    fi
fi

# --- 3. fetch helper (used by --fetch and --fetch-only) -------------------
do_fetch() {
    local pt_cmd=(rsync -az -e "$RSYNC_SSH" --include '*.pt' --include '*.json'
        --exclude '*' "$REMOTE:$REMOTE_DIR/checkpoints/" "$HERE/checkpoints/")
    local csv_cmd=(rsync -az -e "$RSYNC_SSH" --include 'scores*.csv' --exclude '*'
        "$REMOTE:$REMOTE_DIR/" "$HERE/")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ ${pt_cmd[*]}"
        echo "+ ${csv_cmd[*]}"
        return 0
    fi

    mkdir -p "$HERE/checkpoints"
    echo "fetching checkpoints/ and scores*.csv ..."
    "${pt_cmd[@]}" || true
    "${csv_cmd[@]}" || true

    shopt -s nullglob
    local found=("$HERE"/checkpoints/*.pt "$HERE"/checkpoints/history-*.json)
    if [ ${#found[@]} -gt 0 ]; then
        for f in "${found[@]}"; do echo "  $f"; done
    else
        echo "  (no checkpoints yet -- did the job fail? see the slurm output)"
    fi
    local csvs=("$HERE"/scores*.csv)
    for f in ${csvs[@]+"${csvs[@]}"}; do
        echo "  $f ($(($(wc -l < "$f") - 1)) rows)"
    done
}

if [ "$FETCH_ONLY" -eq 1 ]; then
    do_fetch
    exit 0
fi

# --- 4. pick the features folder ------------------------------------------
# Precedence: --features-dir flag > FEATURES_DIR env > the folder that matches
# config.py (or --feature/--pooling when given) > whatever exists.
FEATURES_DIR="${FEATURES_DIR:-}"

ARG_FEATURE=""
ARG_POOLING=""
i=0
n_args=${#JOB_ARGS[@]}
while [ "$i" -lt "$n_args" ]; do
    arg="${JOB_ARGS[$i]}"
    next=""
    if [ $((i + 1)) -lt "$n_args" ]; then next="${JOB_ARGS[$((i + 1))]}"; fi
    case "$arg" in
        --feature)   ARG_FEATURE="$next" ;;
        --feature=*) ARG_FEATURE="${arg#*=}" ;;
        --pooling)   ARG_POOLING="$next" ;;
        --pooling=*) ARG_POOLING="${arg#*=}" ;;
    esac
    i=$((i + 1))
done

if [ -z "$ARG_FEATURE" ] || [ -z "$ARG_POOLING" ]; then
    CFG_VALUES="$(cd "$HERE" && python3 -c 'from config import TrainingConfig; c = TrainingConfig(); print(c.feature, c.pooling)' 2>/dev/null || echo "mfcc mean_std")"
    read -r CFG_FEATURE CFG_POOLING <<< "$CFG_VALUES"
    ARG_FEATURE="${ARG_FEATURE:-${CFG_FEATURE:-mfcc}}"
    ARG_POOLING="${ARG_POOLING:-${CFG_POOLING:-mean_std}}"
fi
WANTED="features/$ARG_FEATURE-$ARG_POOLING"

if [ -z "$FEATURES_DIR" ]; then
    if [ -d "$HERE/$WANTED" ]; then
        FEATURES_DIR="$WANTED"
    else
        for candidate in "$HERE"/features/*/; do
            [ -d "$candidate" ] || continue
            FEATURES_DIR="features/$(basename "$candidate")"
            break
        done
        if [ -n "$FEATURES_DIR" ]; then
            echo "warning: $WANTED not found -- falling back to $FEATURES_DIR" >&2
            echo "         (train.py follows that folder's manifest, not --feature)" >&2
        fi
    fi
fi
if [ -z "$FEATURES_DIR" ]; then
    echo "no features/ folder found -- run 'python prepare_features.py' first" >&2
    exit 1
fi
if [ ! -f "$HERE/$FEATURES_DIR/train.npz" ]; then
    echo "warning: $FEATURES_DIR/train.npz is missing on this machine" >&2
fi

# train.py / evaluate.py must be told where the vectors are, unless you already
# passed --features-dir yourself.
for arg in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
    case "$arg" in
        *[0-9].) echo "warning: '$arg' looks like a typo (trailing dot)" >&2 ;;
    esac
done

HAS_FEATURES_FLAG=0
for arg in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
    case "$arg" in --features-dir|--features-dir=*) HAS_FEATURES_FLAG=1 ;; esac
done
if [ "$HAS_FEATURES_FLAG" -eq 0 ]; then
    JOB_ARGS+=(--features-dir "$FEATURES_DIR")
fi

# --- 5. sync the code + features ------------------------------------------
SYNC_CMD=(rsync -az -e "$RSYNC_SSH"
    --exclude .git --exclude .ssh --exclude __pycache__ --exclude '*.pyc'
    --exclude .DS_Store --exclude cache --exclude checkpoints --exclude logs
    --exclude 'slurm-*.out' --exclude 'scores*.csv' --exclude py --exclude .venv
    "$HERE"/ "$REMOTE:$REMOTE_DIR/")

if [ "$SYNC" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ ${SYNC_CMD[*]}"
    else
        echo "syncing $HERE/ -> $REMOTE:$REMOTE_DIR/ (features: $FEATURES_DIR)"
        "${SYNC_CMD[@]}"
    fi
else
    echo "skipping rsync (--no-sync)"
fi

# --- 6. submit -------------------------------------------------------------
ARGSTR=""
for arg in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
    ARGSTR+="$(printf ' %q' "$arg")"
done
SUBMIT="cd '$REMOTE_DIR' && sbatch --parsable gpu.slurm $TARGET$ARGSTR"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ ssh $REMOTE \"$SUBMIT\""
    echo "(dry run: nothing submitted)"
    exit 0
fi

JID=$("${SSH_CMD[@]}" "$REMOTE" "$SUBMIT" || true)
JID="${JID%%;*}"   # drop any ";clustername" suffix from --parsable
case "$JID" in
    ''|*[!0-9]*)
        echo "sbatch failed: ${JID:-no job id returned}" >&2
        exit 1 ;;
esac

OUT="slurm-$JID.out"
echo "submitted job $JID ($TARGET$ARGSTR)"
echo "  queue : ssh $REMOTE 'squeue -j $JID'"
echo "  output: ssh $REMOTE 'cd $REMOTE_DIR && tail -f $OUT'"
echo "  cancel: ssh $REMOTE 'scancel $JID'"

# --- 7. optionally wait and show the result -------------------------------
if [ "$FOLLOW" -eq 1 ]; then
    echo "waiting for job $JID ..."
    while "${SSH_CMD[@]}" "$REMOTE" "squeue -h -j $JID 2>/dev/null | grep -q ."; do
        sleep 5
    done
    echo "--- $OUT ---"
    "${SSH_CMD[@]}" "$REMOTE" "cd '$REMOTE_DIR' && tail -n 60 '$OUT'"
fi

# --- 8. pull artefacts back ----------------------------------------------
if [ "$FETCH" -eq 1 ]; then
    do_fetch
fi
