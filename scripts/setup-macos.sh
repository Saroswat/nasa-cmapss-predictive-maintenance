#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Saroswat/nasa-cmapss-predictive-maintenance.git"
INSTALL_DIRECTORY="$PWD/nasa-cmapss-predictive-maintenance"
RUN_EXPERIMENT=false
START_DASHBOARD=false

usage() {
  cat <<'EOF'
Install NASA C-MAPSS Predictive Maintenance on macOS.

Usage:
  bash setup-macos.sh [options]

Options:
  --repository-url URL    Git repository to clone
  --install-directory DIR Installation directory
  --run-experiment        Train and evaluate the models after setup
  --start-dashboard       Start the dashboard after setup
  -h, --help              Show this help
EOF
}

while (($#)); do
  case "$1" in
    --repository-url)
      [[ $# -ge 2 ]] || { echo "Missing value for --repository-url" >&2; exit 2; }
      REPOSITORY_URL="$2"
      shift 2
      ;;
    --install-directory)
      [[ $# -ge 2 ]] || { echo "Missing value for --install-directory" >&2; exit 2; }
      INSTALL_DIRECTORY="$2"
      shift 2
      ;;
    --run-experiment)
      RUN_EXPERIMENT=true
      shift
      ;;
    --start-dashboard)
      START_DASHBOARD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null 2>&1 || {
  echo "Git is required. Install the Xcode Command Line Tools with: xcode-select --install" >&2
  exit 1
}

command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 || {
  echo "Node.js 22.13 or newer is required: https://nodejs.org" >&2
  exit 1
}

node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=13)?0:1)' || {
  echo "Node.js 22.13 or newer is required. Found $(node --version)." >&2
  exit 1
}

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  UV_BIN="$(command -v uv || true)"
fi

if [[ -z "$UV_BIN" ]]; then
  echo "uv was installed but could not be found. Open a new Terminal and rerun this script." >&2
  exit 1
fi

if [[ -d "$INSTALL_DIRECTORY/.git" ]]; then
  echo "Updating existing repository..."
  git -C "$INSTALL_DIRECTORY" pull --ff-only
elif [[ -e "$INSTALL_DIRECTORY" ]]; then
  echo "Install path exists but is not a Git repository: $INSTALL_DIRECTORY" >&2
  exit 1
else
  echo "Cloning repository..."
  git clone "$REPOSITORY_URL" "$INSTALL_DIRECTORY"
fi

cd "$INSTALL_DIRECTORY"

echo "Installing Python dependencies..."
"$UV_BIN" sync --extra notebook --extra dev

echo "Downloading and verifying NASA C-MAPSS FD001..."
"$UV_BIN" run cmapss-maintenance download

echo "Installing dashboard dependencies..."
npm --prefix web ci

if [[ "$RUN_EXPERIMENT" == true ]]; then
  echo "Training models and refreshing dashboard data..."
  "$UV_BIN" run cmapss-maintenance run
  "$UV_BIN" run python scripts/export_dashboard_data.py
fi

echo
echo "Setup complete: $INSTALL_DIRECTORY"
echo "Run experiment:  cd \"$INSTALL_DIRECTORY\" && uv run cmapss-maintenance run"
echo "Open notebook:   cd \"$INSTALL_DIRECTORY\" && uv run jupyter lab notebooks/01_modern_predictive_maintenance.ipynb"
echo "Open dashboard:  cd \"$INSTALL_DIRECTORY\" && npm --prefix web run dev"
echo "Dashboard URL:   http://localhost:3000"

if [[ "$START_DASHBOARD" == true ]]; then
  echo
  echo "Starting dashboard..."
  npm --prefix web run dev
fi
