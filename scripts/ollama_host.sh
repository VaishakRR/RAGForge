#!/usr/bin/env bash
# Discover the Windows host gateway from WSL2 and export OLLAMA_BASE_URL.
#
# Usage:
#   source scripts/ollama_host.sh
#
# RAGForge can auto-detect this address, so sourcing is optional.

gateway="$(ip route show default 2>/dev/null | awk '{print $3; exit}')"

if [ -z "$gateway" ]; then
    echo "Could not determine the default gateway." >&2
    return 1 2>/dev/null || exit 1
fi

export OLLAMA_BASE_URL="http://${gateway}:11434"
echo "OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"

if curl -fsS --max-time 5 "${OLLAMA_BASE_URL}/api/version" >/dev/null 2>&1; then
    echo "Ollama is reachable."
else
    echo "Ollama is not reachable. Run scripts/setup_ollama.ps1 in Windows first." >&2
    return 1 2>/dev/null || exit 1
fi
