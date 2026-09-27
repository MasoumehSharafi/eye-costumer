#!/bin/zsh

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eyedropshop-brain

export BRAIN_RETRIEVAL_BACKEND=postgres

echo "Starting EyeDropShop Brain..."
echo "Close this window or press Ctrl+C to stop the Brain."

streamlit run app.py
