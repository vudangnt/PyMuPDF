#!/bin/bash
set -e

SSH="ssh -p 2232 root@222.255.148.172"
REMOTE_DIR="/www/wwwroot/pdf-to-text.digisource.vn"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Syncing files to server..."
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='docs/' \
  --exclude='tests/' \
  --exclude='src_classic/' \
  --exclude='.github/' \
  -e "ssh -p 2232" \
  "$LOCAL_DIR/" \
  "root@222.255.148.172:$REMOTE_DIR/"

echo "==> Installing dependencies..."
$SSH "cd $REMOTE_DIR && pip3 install -r api/requirements.txt -q"

echo "==> Copying supervisor config..."
$SSH "cp $REMOTE_DIR/supervisor/pdf-api.conf /etc/supervisor/conf.d/pdf-api.conf"

echo "==> Reloading supervisor..."
$SSH "supervisorctl reread && supervisorctl update && supervisorctl restart pdf-api || supervisorctl start pdf-api"

echo "==> Checking status..."
$SSH "supervisorctl status pdf-api"

echo ""
echo "Done! API running at http://pdf-to-text.digisource.vn"
echo "Health check: curl http://pdf-to-text.digisource.vn/health"
