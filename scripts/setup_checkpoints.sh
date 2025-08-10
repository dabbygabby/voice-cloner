#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR/OpenVoice"
# V2
TARGET_V2="./checkpoints_v2"
TMP_ZIP=$(mktemp -t ckpts.XXXXXX.zip)
TMP_DIR=$(mktemp -d -t ckpts.XXXXXX)
mkdir -p "$TARGET_V2"
echo "Downloading OpenVoice V2 checkpoints..."
curl -L -o "$TMP_ZIP" "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
unzip -q "$TMP_ZIP" -d "$TMP_DIR"
SRC="$TMP_DIR"
[ -d "$TMP_DIR/checkpoints_v2" ] && SRC="$TMP_DIR/checkpoints_v2"
rsync -a "$SRC"/ "$TARGET_V2"/
rm -f "$TMP_ZIP"
# V1 base EN (fallback)
TARGET_V1="./checkpoints"
V1_ZIP=$(mktemp -t ckpts1.XXXXXX.zip)
V1_DIR=$(mktemp -d -t ckpts1.XXXXXX)
mkdir -p "$TARGET_V1"
echo "Downloading OpenVoice V1 base checkpoints (EN/ZH + converter)..."
curl -L -o "$V1_ZIP" "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
unzip -q "$V1_ZIP" -d "$V1_DIR"
V1_SRC="$V1_DIR"
[ -d "$V1_DIR/checkpoints" ] && V1_SRC="$V1_DIR/checkpoints"
rsync -a "$V1_SRC"/ "$TARGET_V1"/
rm -f "$V1_ZIP"
# Summary
printf "\nInstalled files:\n"
find "$TARGET_V2" -maxdepth 3 -type f | sort | sed "s|^|V2: |"
find "$TARGET_V1" -maxdepth 3 -type f | sort | sed "s|^|V1: |"
