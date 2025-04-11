#!/bin/zsh
# For all PIDs listed
#   pgrep Mathematica or pgrep Wolfram
#   kill -KILL PID

PIDs=$( (pgrep -f Mathematica || true; pgrep -f Wolfram || true) | sort -u )
echo "$PIDs"

if [ -z "$PIDs" ]; then
  echo "no mathematica / wolfram kernels running"
  exit 0
fi

# # else kill all
# for pid in $PIDs; do
#   if [[ "$pid" =~ ^[0-9]+$ ]]; then
#     if kill -KILL "$pid"; then
#       echo "kill signal sent to $pid"
#   fi
# done
pkill -KILL -f Mathematica
pkill -KILL -f Wolfram
echo "all mathematica / wolfram kernels killed"