#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME="test_coffee"
echo "Object name: $OBJ_NAME"
echo "Current work dir: $PROJECT_DIR"
python $PROJECT_DIR/inference_demo.py \
    +experiment="test_demo" \
    input.data_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test" \
    input.sfm_model_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \
    +object_detect_mode="feature_matching"
    use_tracking=True \
    save_wis3d=True
#READ THIS
#The main difference between inference.py and inference_demo.py is that 2D matching has been removed
# in favor of separately running 'feature_matching_object_detector.py' as well as overall code seems a little cleaner. Inference.py also contains visualization tools.
# I would consider abandoning inference_demo.py at all and switching purely to inference.py
# in future interations.