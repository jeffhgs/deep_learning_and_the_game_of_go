#!/bin/bash
dir="$1"
shift
cd "$dir"
exec "$@"
