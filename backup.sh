#!/usr/bin/env bash
USER=ipetruli
SERVER=edgar
FOLDER=/home/clear/ipetruli/projects/bilevel-optimization/src
# --exclude={'wandb', '*.stdout', '*.stderr'}

rsync -Pavu $USER@$SERVER:$FOLDER .
git add -u
git commit -m "daily updates"
git push
