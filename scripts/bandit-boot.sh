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

# Wait for the tailnet address. tailscaled reaching "active" is not the same as
# being logged in and addressed -- observed 2026-09-04, where the service was
# active 3s before dockerd but the address took over two more minutes. Ten
# minutes rather than two, and the timeout says so: the old loop fell through
# silently, and the only clue was a log line that never appeared.
tailscale_ready=no
for _ in $(seq 1 300); do
  if ip -4 addr show tailscale0 2>/dev/null | grep -q 'inet 100\.'; then
    log "tailscale0 has its address"
    tailscale_ready=yes
    break
  fi
  sleep 2
done
[ "$tailscale_ready" = yes ] || log "WARNING: no tailnet address after 10 min -- continuing, supervisor will retry"

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
# before this script runs and before tailscale0 has its address. The api
# container comes back with its port bindings silently dropped -- running,
# passing its healthcheck (which curls itself from inside, where nothing is
# wrong), and publishing nothing. Compose then sees a running container whose
# config matches and leaves it alone.
#
# This was a single check-and-repair before, which is why it failed on
# 2026-09-04: the address was still absent when the one repair ran, the recreate
# failed, and nothing tried again for three days. It is a supervision loop now.
# The loop also replaces the bare `sleep infinity` that held the distro open --
# it never exits, so it does that job too.
log "supervising api port publishing (60s interval)"

last_state=""
note() {  # log only on change, so a long outage is not a line a minute
  [ "$1" = "$last_state" ] && return
  log "$2"
  last_state="$1"
}

while true; do
  if [ -n "$(docker port bandit-api-1 2>/dev/null)" ]; then
    note ok "api ports ok: $(docker port bandit-api-1 | tr '\n' ' ')"
  elif ! ip -4 addr show tailscale0 2>/dev/null | grep -q 'inet 100\.'; then
    note notail "api unpublished and tailscale0 has no address -- waiting for it"
  else
    log "api has no published ports -- recreating"
    if docker compose "${COMPOSE[@]}" up -d --force-recreate api >>"$LOG" 2>&1; then
      log "api recreated: $(docker port bandit-api-1 | tr '\n' ' ')"
      last_state=ok
    else
      note failed "api recreate FAILED -- retrying every 60s"
    fi
  fi
  sleep 60
done
