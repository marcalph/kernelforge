#!/bin/bash
# Deploy kernelforge to tower

REMOTE="tower.local"
REMOTE_PATH="~/projects/kernelforge"

# Sync project (exclude build artifacts and git)
rsync -avz --progress \
    --exclude 'build/' \
    --exclude '.git/' \
    --exclude '*.o' \
    --exclude '*.out' \
    . ${REMOTE}:${REMOTE_PATH}/

echo "Deployed to ${REMOTE}:${REMOTE_PATH}"
