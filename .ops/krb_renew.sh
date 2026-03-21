#!/usr/bin/env bash
set -euo pipefail
INTERVAL_MIN=25
: "${KRB5CCNAME:=}"
if [ -z "${KRB5CCNAME}" ]; then
  export KRB5CCNAME="FILE:/tmp/krb5cc_$(id -u)"
fi
LOG_FILE="/home/tiger/LLM-quickstart/.ops/krb_renew.log"
echo "[start] $(date -Is) using KRB5CCNAME=$KRB5CCNAME" >> "$LOG_FILE"
while true; do
  if kinit -R >> "$LOG_FILE" 2>&1; then
    echo "[ok] $(date -Is) renewed" >> "$LOG_FILE"
  else
    echo "[fail] $(date -Is) renew failed (no renewable ticket?)" >> "$LOG_FILE"
  fi
  sleep $((INTERVAL_MIN*60))
done
