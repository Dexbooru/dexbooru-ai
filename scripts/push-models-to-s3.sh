#!/usr/bin/env bash
# Upload model artifacts to S3 when models/metadata/*.meta.json changed (vs git) or staged.
# Loads AWS credentials from repo-root .env into the environment for aws-cli.
#
# Usage:
#   ./scripts/push-models-to-s3.sh           # working tree vs HEAD + untracked under models/metadata/
#   ./scripts/push-models-to-s3.sh --staged # only staged files (pre-commit style)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUCKET="dexbooru-prd-machine-learning-models"

usage() {
  echo "Usage: $0 [--staged]" >&2
  echo "  --staged   Only consider git staged files under models/metadata/ (pre-commit style)." >&2
}

STAGED_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --staged)
      STAGED_ONLY=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v aws &>/dev/null; then
  echo "error: aws CLI not found; install AWS CLI v2 and ensure it is on PATH." >&2
  exit 1
fi

ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
else
  echo "error: ${ENV_FILE} not found (required for AWS credentials)." >&2
  exit 1
fi

if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree &>/dev/null; then
  echo "error: not a git repository: ${REPO_ROOT}" >&2
  exit 1
fi

collect_meta_paths() {
  local -a lines=()
  if [[ "$STAGED_ONLY" == true ]]; then
    while IFS= read -r line; do
      [[ -n "$line" ]] && lines+=("$line")
    done < <(git -C "$REPO_ROOT" diff --cached --name-only -- models/metadata/ || true)
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] && lines+=("$line")
    done < <(git -C "$REPO_ROOT" diff HEAD --name-only -- models/metadata/ || true)
    while IFS= read -r line; do
      [[ -n "$line" ]] && lines+=("$line")
    done < <(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- models/metadata/ || true)
  fi

  local -a meta=()
  local seen="" path
  for path in "${lines[@]}"; do
    [[ "$path" != *.meta.json ]] && continue
    [[ " $seen " == *" $path "* ]] && continue
    seen+=" $path"
    meta+=("$path")
  done
  printf '%s\n' "${meta[@]}"
}

mapfile -t META_PATHS < <(collect_meta_paths)

if [[ ${#META_PATHS[@]} -eq 0 ]] || [[ -z "${META_PATHS[0]:-}" ]]; then
  echo "Nothing to upload: no added or changed files under models/metadata/ (mode: $(
    [[ "$STAGED_ONLY" == true ]] && echo staged || echo working-tree
  ))."
  exit 0
fi

echo "Uploading for ${#META_PATHS[@]} metadata file(s) to s3://${BUCKET}/"

for rel in "${META_PATHS[@]}"; do
  meta_abs="${REPO_ROOT}/${rel}"
  if [[ ! -f "$meta_abs" ]]; then
    echo "error: metadata file missing on disk: ${rel}" >&2
    exit 1
  fi

  base="$(basename "$rel" .meta.json)"
  skops_rel="models/${base}.skops"
  skops_abs="${REPO_ROOT}/${skops_rel}"
  if [[ ! -f "$skops_abs" ]]; then
    echo "error: expected model next to metadata: ${skops_rel} (from ${rel})" >&2
    exit 1
  fi

  echo "  -> s3://${BUCKET}/${rel}"
  aws s3 cp "$meta_abs" "s3://${BUCKET}/${rel}"

  echo "  -> s3://${BUCKET}/${skops_rel}"
  aws s3 cp "$skops_abs" "s3://${BUCKET}/${skops_rel}"
done

echo "Done."
