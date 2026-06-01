#!/bin/bash
# launch_mega.sh -- start the overnight max-volume scrape as detached workers.
#
# 5 workers on ONE API key, ~0.6s sleep each => ~500 req/min combined,
# under the published 600/min ceiling. Runs under caffeinate so the Mac
# won't sleep mid-scrape. All processes are nohup'd so they survive the
# terminal closing and don't disturb other work.
set -e
cd "$(dirname "$0")"
VENV=/Users/vedvedere/Desktop/Math_168/.venv/bin/python
MEGA=../data/raw/mega
mkdir -p "$MEGA/logs"

# Keep the machine awake for the duration (idle + display sleep prevented).
nohup caffeinate -i > "$MEGA/logs/caffeinate.log" 2>&1 &
echo "caffeinate PID $!"

start() { # mode extra-args logname
  nohup "$VENV" mega_worker.py $1 --sleep 0.6 > "$MEGA/logs/$2.log" 2>&1 &
  echo "  $2 -> PID $!"
}

echo "launching workers:"
start "--mode global-new"                 "global_new"
start "--mode global-top"                 "global_top"
start "--mode submolt-shard --shard 0 --nshards 3" "shard0"
start "--mode submolt-shard --shard 1 --nshards 3" "shard1"
start "--mode submolt-shard --shard 2 --nshards 3" "shard2"
echo "all workers launched. logs in $MEGA/logs/"
