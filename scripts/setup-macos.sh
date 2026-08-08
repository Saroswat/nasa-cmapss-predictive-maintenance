#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-https://github.com/Saroswat/nasa-cmapss-predictive-maintenance.git}"
INSTALL_DIR="${2:-$PWD/nasa-cmapss-predictive-maintenance}"

command -v git >/dev/null 2>&1 || {
  echo "Git is required. Install the Xcode Command Line Tools with: xcode-select --install" >&2
  exit 1
}

command -v npm >/dev/null 2>&1 || {
  echo "Node.js 22.13 or newer is required for the web dashboard: https://nodejs.org" >&2
  exit 1
}

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" pull --ff-only
elif [[ -e "$INSTALL_DIR" ]]; then
  echo "Install path exists but is not a Git repository: $INSTALL_DIR" >&2
  exit 1
else
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
uv sync --extra notebook --extra dev
uv run cmapss-maintenance download
npm --prefix web ci

echo
echo "Setup complete: $INSTALL_DIR"
echo "Run the project: cd \"$INSTALL_DIR\" && uv run cmapss-maintenance run"
echo "Open the notebook: uv run jupyter lab notebooks/01_modern_predictive_maintenance.ipynb"
echo "Open the dashboard: npm --prefix web run dev"
