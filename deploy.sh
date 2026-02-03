#!/bin/bash
# Deploy kernelforge to tower

REMOTE="tower"
REMOTE_PATH="~/kernelforge"

# Sync project (exclude build artifacts and git)
rsync -avz --progress \
    --exclude 'build/' \
    --exclude '.git/' \
    --exclude '*.o' \
    --exclude '*.out' \
    . ${REMOTE}:${REMOTE_PATH}/

echo "Deployed to ${REMOTE}:${REMOTE_PATH}"
