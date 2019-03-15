#!/bin/bash
adirCode="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

exec env PYTHONPATH=${adirCode} "$@"