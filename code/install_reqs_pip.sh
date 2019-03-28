#!/bin/bash
adirCode="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pip3 install -r "$adirCode/requirements-gpu.txt"
