#!/usr/bin/env bash
# Started by the Windows scheduled task "Keep WSL Alive" at logon.
#
# Does two jobs:
#   1. Brings the Bandit stack up once Tailscale is actually ready.
#   2. Never exits, which is what holds the WSL distro open. WSL shuts a distro
#      down when its last client session ends -- running systemd services do NOT
#      keep it alive, so something must stay attached.
#
# Why the wait: the API publishes on this host's tailnet address, and dockerd starts
# before tailscaled has finished assigning that address. Binding an address
# that does not exist yet fails with "cannot assign requested address", exit
# 128 -- and a restart policy does not retry it, because the container failed
# to create its network rather than failing while running.

set -u

# Derived from this script's own location so the checkout can move without the
# scheduled task needing an edit.
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
COMPOSE=(-f docker-compose.yaml -f docker-compose.gpu.yaml -f docker-compose.local.yaml)
LOG=/tmp/bandit-boot.log

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "=== boot script started ==="

# Wait for the tailnet address, up to two minutes.
for _ in $(seq 1 60); do
  if ip -4 addr show tailscale0 2>/dev/null | grep -q 'inet 100\.'; then
    log "tailscale0 has its address"
    break
  fi
  sleep 2
done

# Wait for dockerd to accept connections, up to a minute.
for _ in $(seq 1 30); do
  docker info >/dev/null 2>&1 && break
  sleep 2
done

cd "$REPO" || { log "FATAL: $REPO missing"; exec sleep infinity; }

# Idempotent: no-op when everything is already running and current.
if docker compose "${COMPOSE[@]}" up -d >>"$LOG" 2>&1; then
  log "compose up ok"
else
  log "compose up FAILED -- see above"
fi

# dockerd restores `restart: unless-stopped` containers at boot, which happens
# BEFORE this script runs and before tailscale0 has its address. The api
# container comes back up with its port bindings silently dropped -- running,
# passing its healthcheck (which curls itself from inside), but publishing
# nothing. Compose then sees a running container whose config matches and
# leaves it alone, so waiting for tailscale0 above is necessary but not
# sufficient. Detect the empty mapping and force a real recreate.
if [ -z "$(docker port bandit-api-1 2>/dev/null)" ]; then
  log "api has no published ports -- recreating"
  docker compose "${COMPOSE[@]}" up -d --force-recreate api >>"$LOG" 2>&1 \
    && log "api recreated: $(docker port bandit-api-1 | tr '\n' ' ')" \
    || log "api recreate FAILED"
else
  log "api ports ok: $(docker port bandit-api-1 | tr '\n' ' ')"
fi

log "holding distro open"
exec sleep infinity
