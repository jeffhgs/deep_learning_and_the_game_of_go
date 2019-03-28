#!/bin/bash
adirCode="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
afileScript="$1"
shift

rfileScript=$(basename "$afileScript")
afileStatus="$adirCode/../../.${rfileScript}_success"
if [ ! -e "$afileStatus" ]
then
  chmod +x "$afileScript"
  if "$afileScript"
  then
    touch "$afileStatus"
  else
    touch "${afileStatus}_not"
  fi
fi
exec "$@"