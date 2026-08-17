#!/bin/bash
PROJECT_DIR="$(pwd)"
VIDEO_PATH=${PROJECT_DIR}/data/demo/test_coffee/test_coffee-test

echo '-------------------'
echo 'Parse full image: '
echo '-------------------'

# Parse full image from Frames.m4v
python $PROJECT_DIR/video2img.py \
    --input ${VIDEO_PATH}
    