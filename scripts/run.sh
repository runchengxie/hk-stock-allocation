#!/usr/bin/env bash
set -euo pipefail

uv run hk-alloc --config configs/universe/portfolio.yml "$@"
