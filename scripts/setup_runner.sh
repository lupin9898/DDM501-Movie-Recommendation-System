#!/usr/bin/env bash
# =============================================================================
# setup_runner.sh — Cài đặt GitHub Actions self-hosted runner
#
# Cách dùng:
#   ./scripts/setup_runner.sh \
#     --repo   https://github.com/OWNER/REPO \
#     --token  <REGISTRATION_TOKEN>
#
# Token lấy tại: Settings → Actions → Runners → New self-hosted runner
# =============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RUNNER_VERSION="2.316.1"
RUNNER_DIR="${HOME}/actions-runner"
RUNNER_USER="${USER}"
REPO_URL=""
REG_TOKEN=""
RUNNER_NAME="${HOSTNAME:-$(hostname)}-recsys"
RUNNER_LABELS="self-hosted,Linux,recsys"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 --repo <REPO_URL> --token <REG_TOKEN> [--name <NAME>] [--dir <DIR>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)   REPO_URL="$2";    shift 2 ;;
    --token)  REG_TOKEN="$2";   shift 2 ;;
    --name)   RUNNER_NAME="$2"; shift 2 ;;
    --dir)    RUNNER_DIR="$2";  shift 2 ;;
    *)        usage ;;
  esac
done

[[ -z "$REPO_URL" || -z "$REG_TOKEN" ]] && usage

# ── Detect OS/arch ────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)  RUNNER_ARCH="linux-x64"  ;;
      aarch64) RUNNER_ARCH="linux-arm64" ;;
      *) echo "Unsupported arch: $ARCH"; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64)  RUNNER_ARCH="osx-x64"   ;;
      arm64)   RUNNER_ARCH="osx-arm64" ;;
      *) echo "Unsupported arch: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"; exit 1 ;;
esac

RUNNER_PKG="actions-runner-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PKG}"

echo "=== GitHub Actions Runner Setup ==="
echo "OS/Arch : $OS / $ARCH  →  $RUNNER_ARCH"
echo "Version : $RUNNER_VERSION"
echo "Dir     : $RUNNER_DIR"
echo "Repo    : $REPO_URL"
echo "Name    : $RUNNER_NAME"
echo ""

# ── Install system dependencies (Linux only) ──────────────────────────────────
if [[ "$OS" == "Linux" ]]; then
  echo "→ Installing system dependencies..."
  sudo apt-get update -q
  sudo apt-get install -y -q curl docker.io docker-compose-plugin git
  # Thêm user vào group docker để chạy docker không cần sudo
  sudo usermod -aG docker "$RUNNER_USER" || true
  echo "   docker group: added $RUNNER_USER (re-login to take effect)"
fi

# ── Download runner ───────────────────────────────────────────────────────────
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [[ ! -f "$RUNNER_PKG" ]]; then
  echo "→ Downloading $RUNNER_PKG..."
  curl -fsSL "$RUNNER_URL" -o "$RUNNER_PKG"
else
  echo "→ Runner archive already downloaded — skipping."
fi

echo "→ Extracting..."
tar xzf "$RUNNER_PKG" --overwrite

# ── Configure runner ─────────────────────────────────────────────────────────
echo "→ Configuring runner..."
./config.sh \
  --url "$REPO_URL" \
  --token "$REG_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work "_work" \
  --unattended \
  --replace

# ── Install as system service ─────────────────────────────────────────────────
echo "→ Installing service..."

if [[ "$OS" == "Linux" ]]; then
  # systemd
  sudo ./svc.sh install "$RUNNER_USER"
  sudo ./svc.sh start
  sudo ./svc.sh status
  echo ""
  echo "✓ Runner installed as systemd service."
  echo "  Manage: sudo ./svc.sh {start|stop|status|uninstall}"

elif [[ "$OS" == "Darwin" ]]; then
  # launchd (macOS)
  ./svc.sh install
  ./svc.sh start
  echo ""
  echo "✓ Runner installed as launchd service."
  echo "  Manage: ./svc.sh {start|stop|status|uninstall}"
fi

# ── Post-install notes ────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify runner shows as 'Idle' at:"
echo "     ${REPO_URL}/settings/actions/runners"
echo ""
echo "  2. Set GitHub Secrets (Settings → Secrets → Actions):"
echo "     GRAFANA_PASSWORD   — mật khẩu Grafana (tùy chọn)"
echo ""
echo "  3. Tạo môi trường 'production' (Settings → Environments)"
echo "     và thêm protection rules nếu cần."
echo ""
echo "  4. Chạy deploy thủ công lần đầu:"
echo "     Actions → Deploy → Run workflow"
