#!/bin/bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" = "main" ]; then
  PYTHONPATH=. "/Users/dougstrouth/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/heart_disease/.venv/bin/python" -m pytest
fi
