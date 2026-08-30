#!/usr/bin/env bash
# ==============================================================================
# remote-safeguard.sh
#
# Run Claude Code with `claude --resume` and keep the session alive across
# rate-limit interruptions. If Claude exits because a usage/rate limit was hit
# (e.g. HTTP 429, "limit reached", "Retry-After"), this wrapper:
#   1. detects the rate-limit condition from the exit code and/or output,
#   2. extracts the reset time ("resets at X") or retry-after duration,
#   3. sleeps for exactly that long (plus a small safety buffer),
#   4. loops and re-runs `claude --resume` to continue the same session.
#
# It RESPECTS the limit — it waits out the full cooldown, it does not try to
# bypass or hammer it. Any extra CLI args you pass are forwarded to claude:
#   ./remote-safeguard.sh                 # plain: claude --resume
#   ./remote-safeguard.sh <session-id>    # resume a specific session
#   ./remote-safeguard.sh --model opus    # forward flags through
#
# Tunable via environment variables (all optional):
#   CLAUDE_BIN            binary to run                      (default: claude)
#   MAX_RETRIES          max resume attempts, 0 = unlimited  (default: 0)
#   DEFAULT_WAIT         fallback sleep, seconds             (default: 300)
#   RESET_BUFFER         extra seconds added after reset     (default: 15)
#   MIN_WAIT             floor on any sleep, seconds         (default: 5)
#   MAX_WAIT            ceiling on any sleep, seconds        (default: 86400)
#   RESTART_ON_ANY_ERROR restart on non-rate-limit errors too (0/1, default: 0)
#   LOG_FILE             append run log here                 (default: ./remote-safeguard.log)
# ==============================================================================

set -uo pipefail

# ------------------------------- configuration --------------------------------
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
MAX_RETRIES="${MAX_RETRIES:-0}"
DEFAULT_WAIT="${DEFAULT_WAIT:-300}"
RESET_BUFFER="${RESET_BUFFER:-15}"
MIN_WAIT="${MIN_WAIT:-5}"
MAX_WAIT="${MAX_WAIT:-86400}"
RESTART_ON_ANY_ERROR="${RESTART_ON_ANY_ERROR:-0}"
LOG_FILE="${LOG_FILE:-./remote-safeguard.log}"

# ------------------------------- housekeeping ---------------------------------
CAPTURE_FILE="$(mktemp "${TMPDIR:-/tmp}/remote-safeguard.XXXXXX")"
cleanup() { rm -f "$CAPTURE_FILE"; }
trap cleanup EXIT

INTERRUPTED=0
on_signal() {
  INTERRUPTED=1
  log "Received interrupt — stopping safeguard loop."
  exit 130
}
trap on_signal INT TERM

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE" >&2
}

# Clamp $1 into [MIN_WAIT, MAX_WAIT].
clamp_wait() {
  local w="$1"
  [[ "$w" =~ ^[0-9]+$ ]] || w="$DEFAULT_WAIT"
  (( w < MIN_WAIT )) && w="$MIN_WAIT"
  (( w > MAX_WAIT )) && w="$MAX_WAIT"
  printf '%s' "$w"
}

# ------------------------- rate-limit detection -------------------------------
# Return 0 (true) if the captured output / exit code looks like a rate limit.
is_rate_limited() {
  local rc="$1" out="$2"
  # Claude/Anthropic and generic HTTP signals. Case-insensitive.
  if grep -qiE '429|rate[ _-]?limit|too many requests|usage limit|limit reached|limit exceeded|quota (exceeded|reached)|retry[ _-]?after|resets? (at|in)|try again (at|in|later)|overloaded|529' <<<"$out"; then
    return 0
  fi
  # Some builds use a dedicated exit code for limit conditions.
  case "$rc" in
    7|75|429) return 0 ;;
  esac
  return 1
}

# ---------------------- duration / timestamp parsing --------------------------
# Convert a free-form duration like "2h 5m", "90 minutes", "300s" to seconds.
# Echoes the total seconds, or nothing if it parses to zero/none.
parse_duration_to_seconds() {
  local s; s="$(tr 'A-Z' 'a-z' <<<"$1")"
  local total=0 n
  n="$(grep -oE '[0-9]+ ?h(ou)?r?s?' <<<"$s" | grep -oE '[0-9]+' | head -1)"; [[ -n "$n" ]] && (( total += n*3600 ))
  n="$(grep -oE '[0-9]+ ?m(in(ute)?s?)?' <<<"$s" | grep -oE '[0-9]+' | head -1)"; [[ -n "$n" ]] && (( total += n*60 ))
  n="$(grep -oE '[0-9]+ ?s(ec(ond)?s?)?' <<<"$s" | grep -oE '[0-9]+' | head -1)"; [[ -n "$n" ]] && (( total += n ))
  (( total > 0 )) && printf '%s' "$total"
}

# Turn a clock time / timestamp string into "seconds from now" using GNU date.
# Handles times that already passed today by rolling to tomorrow.
timestamp_to_wait() {
  local when="$1" epoch now
  epoch="$(date -d "$when" +%s 2>/dev/null)" || return 1
  now="$(date +%s)"
  if (( epoch <= now )); then
    # Likely a bare clock time (e.g. "11:00pm") already past today.
    epoch="$(date -d "$when tomorrow" +%s 2>/dev/null)" || return 1
    (( epoch <= now )) && return 1
  fi
  printf '%s' "$(( epoch - now ))"
}

# Inspect captured output and decide how long to wait. Always echoes a number.
compute_wait_seconds() {
  local out="$1" val ph

  # (A) Explicit Retry-After: <seconds>  (header or JSON: retry_after / retryAfter)
  val="$(grep -oiE 'retry[ _-]?after[^0-9]{0,6}[0-9]+' <<<"$out" | grep -oE '[0-9]+' | tail -1)"
  if [[ -n "$val" ]]; then log "Parsed retry-after: ${val}s"; printf '%s' "$val"; return; fi

  # (B) "try again in ...", "resets in ...", "available in 2h 5m"
  ph="$(grep -oiE '(reset[s]?|available|try again|back)[^.]{0,4}in [0-9][0-9 hmsinutecod]*' <<<"$out" | head -1)"
  if [[ -n "$ph" ]]; then
    val="$(parse_duration_to_seconds "${ph#*in }")"
    if [[ -n "$val" ]]; then log "Parsed relative duration from: \"$ph\" -> ${val}s"; printf '%s' "$val"; return; fi
  fi

  # (C) "resets at <time>", "try again at <time>", "available again at <time>"
  ph="$(grep -oiE '(reset[s]?|available( again)?|try again|back)[^.]{0,6}(at|by) [^.\n]+' <<<"$out" | head -1)"
  if [[ -n "$ph" ]]; then
    local when="${ph##* at }"; [[ "$when" == "$ph" ]] && when="${ph##* by }"
    when="$(sed -E 's/[[:space:]]*$//' <<<"$when")"
    val="$(timestamp_to_wait "$when")"
    if [[ -n "$val" ]]; then log "Parsed reset time \"$when\" -> ${val}s from now"; printf '%s' "$val"; return; fi
  fi

  # (D) Fallback.
  log "Could not parse a reset time; using DEFAULT_WAIT=${DEFAULT_WAIT}s"
  printf '%s' "$DEFAULT_WAIT"
}

# ----------------------------- run claude once --------------------------------
# Runs `claude --resume "$@"` keeping the interactive TUI intact while also
# capturing its output to $CAPTURE_FILE for parsing. Returns claude's exit code.
run_claude() {
  : > "$CAPTURE_FILE"
  local rc
  if command -v script >/dev/null 2>&1 && script --version 2>&1 | grep -qi 'util-linux'; then
    # util-linux `script` gives claude a real PTY (so the UI renders normally)
    # and tees the session to a file. -e returns the command's exit status.
    local cmd; cmd="$(printf '%q ' "$CLAUDE_BIN" --resume "$@")"
    script -q -e -c "$cmd" "$CAPTURE_FILE"
    rc=$?
  else
    # Fallback: pipe through tee (loses PTY niceties but still works/captures).
    "$CLAUDE_BIN" --resume "$@" 2>&1 | tee "$CAPTURE_FILE"
    rc=${PIPESTATUS[0]}
  fi
  return "$rc"
}

# --------------------------------- main loop ----------------------------------
log "=== remote-safeguard starting (bin=$CLAUDE_BIN, max_retries=$MAX_RETRIES) ==="

attempt=0
while :; do
  attempt=$(( attempt + 1 ))
  log "Launch attempt #$attempt: $CLAUDE_BIN --resume $*"

  run_claude "$@"
  rc=$?
  out="$(cat "$CAPTURE_FILE" 2>/dev/null || true)"

  (( INTERRUPTED )) && exit 130

  if (( rc == 0 )); then
    log "Claude exited cleanly (code 0). Done."
    exit 0
  fi

  if is_rate_limited "$rc" "$out"; then
    wait_s="$(clamp_wait "$(compute_wait_seconds "$out")")"
    wait_s=$(( wait_s + RESET_BUFFER ))
    wake="$(date -d "+${wait_s} seconds" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || echo "in ${wait_s}s")"
    log "Rate limit detected (exit $rc). Sleeping ${wait_s}s (incl. ${RESET_BUFFER}s buffer); will resume at ${wake}."
  elif (( RESTART_ON_ANY_ERROR == 1 )); then
    wait_s=$(( DEFAULT_WAIT + RESET_BUFFER ))
    log "Non-zero exit ($rc), RESTART_ON_ANY_ERROR=1. Sleeping ${wait_s}s then resuming."
  else
    log "Claude exited with code $rc and it is not a rate limit. Not restarting."
    log "(Set RESTART_ON_ANY_ERROR=1 to auto-resume on other errors.)"
    exit "$rc"
  fi

  if (( MAX_RETRIES > 0 && attempt >= MAX_RETRIES )); then
    log "Reached MAX_RETRIES=$MAX_RETRIES. Giving up."
    exit "$rc"
  fi

  # Sleep in short chunks so Ctrl-C stays responsive during long cooldowns.
  remaining="$wait_s"
  while (( remaining > 0 )); do
    (( INTERRUPTED )) && exit 130
    chunk=$(( remaining > 30 ? 30 : remaining ))
    sleep "$chunk"
    remaining=$(( remaining - chunk ))
  done

  log "Cooldown elapsed. Resuming session…"
done
