#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME="test_coffee"
echo $OBJ_NAME
PYTHONPATH=$PYTHONPATH:$pwd/DeepLM/build
echo "Current work dir: $PROJECT_DIR"


python $PROJECT_DIR/run.py \
    +preprocess="sfm_spp_spg_demo" \
    dataset.data_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-annotate" \
    dataset.outputs_dir="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \