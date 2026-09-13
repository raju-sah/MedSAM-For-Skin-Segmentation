#!/usr/bin/env bash
# ==============================================================================
# CG-MedSAM: Live Public Interactive Demo Launcher
# 
# Launches:
# 1. Local FastAPI + Uvicorn server on port 7860 (if not already running)
# 2. Secure Cloudflare Tunnel to expose port 7860 to the public web with HTTPS
# ==============================================================================

set -e

PORT=7860
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${PROJECT_DIR}/scratch"
CLOUDFLARED_BIN="${SCRATCH_DIR}/cloudflared"

echo "======================================================================"
echo " Starting CG-MedSAM Interactive Clinical Web Demo"
echo "======================================================================"

# 1. Ensure local FastAPI backend is active
if ! curl -s "http://127.0.0.1:${PORT}/api/status" > /dev/null 2>&1; then
    echo "[+] Starting local FastAPI server on port ${PORT}..."
    cd "${PROJECT_DIR}"
    nohup uvicorn web_demo.app:app --host 127.0.0.1 --port ${PORT} > "${PROJECT_DIR}/web_demo_server.log" 2>&1 &
    sleep 3
    if curl -s "http://127.0.0.1:${PORT}/api/status" > /dev/null 2>&1; then
        echo "[✓] Local server is healthy and responding on http://127.0.0.1:${PORT}"
    else
        echo "[!] Warning: Server is still initializing. Check web_demo_server.log"
    fi
else
    echo "[✓] Local server is already running on http://127.0.0.1:${PORT}"
fi

# 2. Check cloudflared binary
if [ ! -f "${CLOUDFLARED_BIN}" ]; then
    mkdir -p "${SCRATCH_DIR}"
    echo "[+] Downloading cloudflared binary..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "${CLOUDFLARED_BIN}"
    chmod +x "${CLOUDFLARED_BIN}"
fi

# 3. Launch or inspect tunnel
echo "[+] Checking Cloudflare Tunnel status..."
TUNNEL_LOG="${PROJECT_DIR}/scratch/tunnel.log"

if pgrep -f "cloudflared tunnel.*${PORT}" > /dev/null 2>&1; then
    echo "[✓] Cloudflare tunnel is already active."
else
    echo "[+] Establishing new Cloudflare quick tunnel..."
    nohup "${CLOUDFLARED_BIN}" tunnel --url "http://127.0.0.1:${PORT}" > "${TUNNEL_LOG}" 2>&1 &
    sleep 8
fi

# 4. Extract public URL
PUBLIC_URL=$(grep -oE "https://[a-zA-Z0-9-]+\.trycloudflare\.com" "${TUNNEL_LOG}" 2>/dev/null | tail -n 1 || true)

echo "======================================================================"
echo " LIVE PUBLIC ACCESS READY"
echo "======================================================================"
if [ -n "${PUBLIC_URL}" ]; then
    echo "  🌐 Public HTTPS URL: ${PUBLIC_URL}"
    echo "  💻 Local URL:        http://127.0.0.1:${PORT}"
    echo "  📊 Health Check:     ${PUBLIC_URL}/api/status"
else
    echo "  Tunnel started. Check log for URL: ${TUNNEL_LOG}"
fi
echo "======================================================================"
