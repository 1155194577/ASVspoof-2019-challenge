#!/usr/bin/env bash
# Deploy + submit for the CUHK CSE GPU cluster (login node linux1).
# Run this from your Mac with the CSE VPN connected.
#
#   ./submit.sh                                  # train.py, defaults from config.py
#   ./submit.sh --epochs 10 --lr 0.02            # flags are forwarded to train.py
#   ./submit.sh --job eval --split validation    # evaluate.py instead
#   ./submit.sh --follow --epochs 3              # submit, wait, print the output
#   ./submit.sh --fetch-only                     # pull experiment_log/ back only
#   ./submit.sh --dry-run --epochs 3             # show commands, touch nothing
#
# Consumed here (everything else is forwarded to the remote script):
#   -j, --job train|eval  remote entry point (default: train)
#   -f, --follow          wait for the job, then tail its slurm output
#   --fetch               copy experiment_log/ (and scores*.csv) back after submitting
#   --fetch-only          only pull experiment_log/ back; submit nothing
#   --no-sync             skip the rsync, submit what is already on the cluster
#   --dry-run             print the rsync/ssh/sbatch commands and exit
#   -h, --help            this text
#
# Env overrides:
#   REMOTE       ssh target              (default: cyho2@linux9, a submit node)
#   SLURM_CONF   remote slurm config      (default: /opt1/slurm/gpu-slurm.conf)
#   REMOTE_DIR   project dir on cluster  (default: aist4010_project)
#   SSH_KEY      private key             (default: ./.ssh/id_ed25519)
#   FEATURES_DIR feature folder to use   (default: the one under ./features)
#
# Only features/ + the source files are synced (~10 MB); the dataset and the
# featurisation cache stay on this machine, which is the point of
# prepare_features.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# linux1 is NOT a submit node: the GPGPU cluster's submit nodes are linux5-16
REMOTE="${REMOTE:-cyho2@linux9}"
SLURM_CONF_REMOTE="${SLURM_CONF:-/opt1/slurm/gpu-slurm.conf}"
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

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes
          -o StrictHostKeyChecking=accept-new)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

# --- 2. is the cluster reachable and do we authenticate? ------------------
if [ "$DRY_RUN" -eq 0 ]; then
    SSH_ERR="$("${SSH_CMD[@]}" "$REMOTE" \
        "SLURM_CONF='$SLURM_CONF_REMOTE' scontrol ping 2>&1" 2>&1 >/dev/null)" || {
        echo "cannot ssh to $REMOTE" >&2
        [ -n "$SSH_ERR" ] && echo "  $SSH_ERR" >&2
        case "$SSH_ERR" in
            *"Permission denied"*)
                echo "  -> the server rejected key $KEY." >&2
                echo "     re-add it:  ssh-copy-id -i ${KEY}.pub $REMOTE" >&2
                echo "     (or check that your cluster account is still active)" >&2 ;;
            *"timed out"*|*"Operation timed out"*|*"Could not resolve"*|*"No route to host"*)
                echo "  -> connect to the CSE VPN first (the cluster only accepts" >&2
                echo "     clients on the campus/VPN network)" >&2 ;;
            *"scontrol"*|*"slurm_load"*|*"ClusterName"*|*"s_p_parse_file"*)
                echo "  -> slurm is not usable on $REMOTE with SLURM_CONF=$SLURM_CONF_REMOTE" >&2
                echo "     submit nodes are linux5-16; try REMOTE=cyho2@linux5" >&2 ;;
        esac
        exit 1
    }
fi

# --- 3. fetch helper (used by --fetch and --fetch-only) -------------------
do_fetch() {
    local csv_cmd=(rsync -az -e "$RSYNC_SSH" --include 'scores*.csv' --exclude '*'
        "$REMOTE:$REMOTE_DIR/" "$HERE/")
    local exp_cmd=(rsync -az -e "$RSYNC_SSH"
        "$REMOTE:$REMOTE_DIR/experiment_log/" "$HERE/experiment_log/")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ ${exp_cmd[*]}"
        echo "+ ${csv_cmd[*]}"
        return 0
    fi

    mkdir -p "$HERE/experiment_log"
    echo "fetching experiment_log/ and scores*.csv ..."
    "${exp_cmd[@]}" || true
    "${csv_cmd[@]}" || true

    if [ -d "$HERE/experiment_log" ]; then
        for f in "$HERE"/experiment_log/exp_*/; do
            [ -d "$f" ] || continue
            echo "  ${f%/}"
        done
    else
        echo "  (no experiments yet -- did the job fail? see the slurm output)"
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
ARG_FEATURES_DIR=""
i=0
n_args=${#JOB_ARGS[@]}
while [ "$i" -lt "$n_args" ]; do
    arg="${JOB_ARGS[$i]}"
    next=""
    if [ $((i + 1)) -lt "$n_args" ]; then next="${JOB_ARGS[$((i + 1))]}"; fi
    case "$arg" in
        --feature)       ARG_FEATURE="$next" ;;
        --feature=*)     ARG_FEATURE="${arg#*=}" ;;
        --pooling)       ARG_POOLING="$next" ;;
        --pooling=*)     ARG_POOLING="${arg#*=}" ;;
        --features-dir)  ARG_FEATURES_DIR="$next" ;;
        --features-dir=*) ARG_FEATURES_DIR="${arg#*=}" ;;
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

# an explicit --features-dir wins over both the env var and config.py
if [ -n "$ARG_FEATURES_DIR" ]; then
    FEATURES_DIR="$ARG_FEATURES_DIR"
fi

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
if [ ! -d "$HERE/$FEATURES_DIR" ]; then
    echo "warning: $FEATURES_DIR does not exist locally (only the cluster copy will be used)" >&2
elif [ ! -f "$HERE/$FEATURES_DIR/train.npz" ]; then
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

# --- 4b. allocate the experiment folder locally ---------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    EXP_DIR="$(cd "$HERE" && python3 -c 'import experiments; print(experiments.peek_next())' 2>/dev/null || echo experiment_log/exp_NNN)"
else
    EXP_DIR="$(cd "$HERE" && python3 -c 'import experiments; print(experiments.create())' 2>/dev/null || true)"
    if [ -z "$EXP_DIR" ]; then
        echo "could not allocate an experiment folder (python3 + experiments.py)" >&2
        exit 1
    fi
fi
JOB_ARGS+=(--exp-dir "$EXP_DIR")
echo "experiment  : $EXP_DIR  (config.json, history.json, model.pt, plot.png)"

# --- 5. sync the code + features ------------------------------------------
# code only: features/ is synced separately below, so a run never pushes
# 1.1 GB of flat features (or exceeds the 2.44 GB home quota) unless it uses them
SYNC_CMD=(rsync -az -e "$RSYNC_SSH"
    --exclude .git --exclude .ssh --exclude __pycache__ --exclude '*.pyc'
    --exclude .DS_Store --exclude cache --exclude checkpoints --exclude logs
    --exclude 'slurm-*.out' --exclude 'scores*.csv' --exclude py --exclude .venv
    --exclude experiment_log --exclude features
    "$HERE"/ "$REMOTE:$REMOTE_DIR/")
# --partial-dir gives a deterministic name, so an interrupted transfer resumes
# where it stopped. (macOS ships openrsync/2.6.9, which rejects --append-verify.)
FEAT_CMD=(rsync -az --partial-dir=.rsync-partial --timeout=120 -e "$RSYNC_SSH"
    "$HERE/$FEATURES_DIR/" "$REMOTE:$REMOTE_DIR/$FEATURES_DIR/")

if [ "$SYNC" -eq 1 ]; then
    echo "syncing code   : $HERE/ -> $REMOTE:$REMOTE_DIR/"
    echo "syncing features: $FEATURES_DIR ($(du -sh "$HERE/$FEATURES_DIR" 2>/dev/null | cut -f1))"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ ${SYNC_CMD[*]}"
        echo "+ ${FEAT_CMD[*]}"
    else
        "${SYNC_CMD[@]}"
        for attempt in 1 2 3; do
            if "${FEAT_CMD[@]}"; then break; fi
            echo "  features sync interrupted (attempt $attempt/3), resuming ..." >&2
            [ "$attempt" -eq 3 ] && { echo "features sync failed" >&2; exit 1; }
            sleep 5
        done
    fi
else
    echo "skipping rsync (--no-sync)"
fi

# --- 6. submit -------------------------------------------------------------
ARGSTR=""
for arg in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
    ARGSTR+="$(printf ' %q' "$arg")"
done
SUBMIT="cd '$REMOTE_DIR' && SLURM_CONF='$SLURM_CONF_REMOTE' sbatch --parsable gpu.slurm $TARGET$ARGSTR"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ ssh $REMOTE \"$SUBMIT\""
    echo "(dry run: nothing submitted)"
    exit 0
fi

JID=$("${SSH_CMD[@]}" "$REMOTE" "$SUBMIT" || true)
case "$JID" in
    *s_p_parse_file*|*"ClusterName needs to be specified"*|*"Unable to process configuration file"*|*"slurm_load_ctl_conf"*)
        echo "sbatch failed: the Slurm controller on $REMOTE looks broken" >&2
        echo "  ($JID)" >&2
        echo "  e.g. /etc/slurm/slurm.conf missing -- server-side, wait for the admin" >&2
        rmdir "$HERE/$EXP_DIR" 2>/dev/null && echo "  (removed unused $EXP_DIR)"
        exit 1 ;;
esac
JID="${JID%%;*}"   # drop any ";clustername" suffix from --parsable
case "$JID" in
    ''|*[!0-9]*)
        echo "sbatch failed: ${JID:-no job id returned}" >&2
        rmdir "$HERE/$EXP_DIR" 2>/dev/null && echo "  (removed unused $EXP_DIR)"
        exit 1 ;;
esac

OUT="slurm-$JID.out"
echo "submitted job $JID ($TARGET$ARGSTR)"
echo "experiment  : $EXP_DIR"
echo "  queue : ssh $REMOTE 'SLURM_CONF=$SLURM_CONF_REMOTE squeue -j $JID'"
echo "  output: ssh $REMOTE 'tail -f $REMOTE_DIR/$OUT'"
echo "  cancel: ssh $REMOTE 'SLURM_CONF=$SLURM_CONF_REMOTE scancel $JID'"

# --- 7. optionally wait and show the result -------------------------------
if [ "$FOLLOW" -eq 1 ]; then
    echo "waiting for job $JID ..."
    while "${SSH_CMD[@]}" "$REMOTE" \
        "SLURM_CONF='$SLURM_CONF_REMOTE' squeue -h -j $JID 2>/dev/null | grep -q ."; do
        sleep 5
    done
    echo "--- $OUT ---"
    "${SSH_CMD[@]}" "$REMOTE" "cd '$REMOTE_DIR' && tail -n 60 '$OUT'"
    mkdir -p "$HERE/$EXP_DIR"
    "${SSH_CMD[@]}" "$REMOTE" "cd '$REMOTE_DIR' && cat '$OUT'" \
        > "$HERE/$EXP_DIR/slurm.out" 2>/dev/null || true
    echo "(slurm output saved to $EXP_DIR/slurm.out)"
fi

# --- 8. pull artefacts back ----------------------------------------------
if [ "$FETCH" -eq 1 ]; then
    do_fetch
fi
