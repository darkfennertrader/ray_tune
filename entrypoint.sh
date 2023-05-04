#!/bin/bash

set -e
echo "pippo"
ray start --head --dashboard-host 0.0.0.0
tail -f /dev/null
