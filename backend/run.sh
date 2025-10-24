#!/bin/zsh
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PREFERRED_DEVICE=${PREFERRED_DEVICE:-}
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
