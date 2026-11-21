#!/usr/bin/env bash
#
# Publish RAGForge to GitHub as a series of small, logical commits.
#
# Instead of one giant initial commit, this script walks the project in
# dependency order (tooling -> core -> store -> retrieval -> generation ->
# API -> CLI -> UI -> tests -> docs -> samples) and pushes after every commit,
# so the remote history fills in a little at a time.
#
# Usage:
#   scripts/publish.sh                 # create VaishakRR/RAGForge (public)
#   scripts/publish.sh --private       # create it private instead
#   scripts/publish.sh --name MyRepo   # use a different repository name
#   scripts/publish.sh --dry-run       # show the plan, change nothing
#   scripts/publish.sh --yes           # skip the confirmation prompt
#
# Safe to re-run: stages with nothing new are skipped, so an interrupted
# publish can be resumed by running the script again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPO_NAME="RAGForge"
VISIBILITY="public"
ASSUME_YES=0
DRY_RUN=0

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --private) VISIBILITY="private"; shift ;;
    --public)  VISIBILITY="public";  shift ;;
    --name)    REPO_NAME="${2:?--name needs a value}"; shift 2 ;;
    --yes|-y)  ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

command -v git >/dev/null || die "git is not installed"
command -v gh  >/dev/null || die "GitHub CLI (gh) is not installed"
gh auth status >/dev/null 2>&1 || die "run 'gh auth login' first"

OWNER="$(gh api user -q .login)"
REPO="$OWNER/$REPO_NAME"

# Git needs an identity to create commits.
if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
  die "set a git identity first: git config --global user.name '...' && git config --global user.email '...'"
fi

# Refuse to publish anything that looks like a secret or an oversized blob.
find . -path ./.git -prune -o -path ./.venv -prune -o -type f \
  \( -name '.env' -o -name '*.pem' -o -name '*.key' -o -name 'id_rsa*' \) -print \
  | grep -q . && die "refusing to publish: found secret-like files (e.g. .env, *.pem, *.key)"

while IFS= read -r -d '' big; do
  die "refusing to publish: $big is larger than 90 MB"
done < <(find . -path ./.git -prune -o -path ./.venv -prune -o -type f -size +90M -print0)

if [[ ! -d .git ]]; then
  echo "Initializing git repository..."
  git init -q -b main
fi
git symbolic-ref HEAD refs/heads/main

if git remote get-url origin >/dev/null 2>&1; then
  echo "Using existing origin: $(git remote get-url origin)"
else
  if gh repo view "$REPO" >/dev/null 2>&1; then
    echo "Repository $REPO already exists; attaching remote."
  elif [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would create $REPO ($VISIBILITY)"
  else
    echo "Creating $REPO ($VISIBILITY)..."
    gh repo create "$REPO" "--$VISIBILITY" \
      --description "Fully local RAG studio: hybrid retrieval, citations, streaming UI, Ollama." \
      >/dev/null
  fi
  [[ $DRY_RUN -eq 1 ]] || git remote add origin "https://github.com/$REPO.git"
fi

# A fresh repo must not already contain history we would clobber.
if [[ $DRY_RUN -eq 0 ]] && ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  if [[ -n "$(git ls-remote --heads origin 2>/dev/null || true)" ]]; then
    die "remote $REPO already has commits; clone it or use --name with a fresh repo"
  fi
fi

echo
echo "Repository : $REPO ($VISIBILITY)"
echo "Branch     : main"
echo "Stages     : 15 logical commits, pushed one at a time"
echo

if [[ $DRY_RUN -eq 0 && $ASSUME_YES -eq 0 ]]; then
  read -r -p "Proceed? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

publish_stage() {
  local message="$1"; shift
  local -a paths=("$@") existing=()
  local p

  for p in "${paths[@]}"; do
    [[ -e "$p" ]] && existing+=("$p")
  done
  [[ ${#existing[@]} -eq 0 ]] && return 0

  if [[ $DRY_RUN -eq 1 ]]; then
    if [[ -n "$(git status --porcelain -- "${existing[@]}")" ]]; then
      echo "[dry-run] commit: $message"
      printf '    %s\n' "${existing[@]}"
    else
      echo "skip: $message (nothing new)"
    fi
    return 0
  fi

  git add -- "${existing[@]}"
  if git diff --cached --quiet; then
    echo "skip: $message (nothing new)"
    return 0
  fi

  git commit -q -m "$message"
  git push -q -u origin main
  echo "pushed: $message"
}

publish_stage "chore: project scaffolding and tooling" \
  pyproject.toml uv.lock .gitignore .env.example Makefile LICENSE

publish_stage "feat(config): environment-driven settings" \
  ragforge/__init__.py ragforge/__main__.py ragforge/config.py

publish_stage "feat(ollama): async client with endpoint discovery" \
  ragforge/ollama_client.py

publish_stage "feat(parsing): extract text from pdf, docx, html, markdown" \
  ragforge/parsing.py

publish_stage "feat(chunking): boundary-aware sliding window" \
  ragforge/chunking.py

publish_stage "feat(embeddings): Ollama embedder with offline fallback" \
  ragforge/embeddings.py

publish_stage "feat(store): sqlite-backed document and vector store" \
  ragforge/store.py

publish_stage "feat(retrieval): dense + bm25 fused with rrf, diversified with mmr" \
  ragforge/retrieval.py

publish_stage "feat(rag): grounded prompt, context selection, streaming answers" \
  ragforge/rag.py ragforge/ingest.py

publish_stage "feat(api): fastapi app with uploads and sse chat" \
  ragforge/api.py

publish_stage "feat(cli): ingest, ask, list, check, reset" \
  ragforge/cli.py

publish_stage "feat(ui): drag-and-drop interface with live citations" \
  ragforge/static

publish_stage "test: cover chunking, retrieval, store, and ingestion" \
  tests

publish_stage "docs: readme, design notes, and deep dive" \
  README.md PLAN.md docs

publish_stage "chore: demo corpus and ollama setup scripts" \
  samples scripts

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Re-run without --dry-run to publish."
  exit 0
fi

echo "Published: https://github.com/$REPO"

untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
  echo
  echo "Note: these files were not part of any stage (check .gitignore or add a stage):"
  printf '  %s\n' "$untracked"
fi
