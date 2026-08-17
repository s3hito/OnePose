#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME="test_coffee"
echo $OBJ_NAME
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:
python $PROJECT_DIR/parse_scanned_data.py \
    --scanned_object_path \
    "$PROJECT_DIR/data/demo/$OBJ_NAME"

echo '--------------------------------------------------------------'
echo 'Run SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from $OBJ_NAME-annotate sequence:
  python $PROJECT_DIR/run.py \
      +preprocess="sfm_spp_spg_demo" \
      dataset.data_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-annotate" \
      dataset.outputs_dir="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \
#Setting dataset.data_dir instead of manually setting in sfm_spp_spg_demo.yaml
#The main difference between inference.py and inference_demo.py is that 2D matching has been removed
# in favor of separately running 'feature_matching_object_detector.py' as well as overall code seems a little cleaner. Inference.py also contains visualization tools.
# I would consider abandoning inference_demo.py at all and switching purely to inference.py
# in future interations.
# dubbing in inference_solo.sh as well