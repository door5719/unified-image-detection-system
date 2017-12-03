# README

$ROOT refers to the unified-detection-system root directory.

$baitcam refers to baitcam dataset directory.


### Install frameworks:

1. Refer to $ROOT/INSTALL.md

### Dataset preparation:
1. Any dataset used with the Unified Detection System should follow the initial structure of Baitcam from the thesis:

  ```Shell
  $baitcam/images               # Contains images
  $baitcam/pascal_Annotations   # Contains .xml annotations (pascal format)
  $baitcam/pascal_Imagesets     # Contains train.txt and val.txt imagesets (pascal format)
  $baitcam/baitcam_labelmap.prototxt # Contains labels and class names
  ```

2. Convert to SSD data format:

  ```Shell
  cd $ROOT/data/data_utils
  python convert_pascal_2_ssd.py -h     # To see arguments
  python convert_pascal_2_ssd.py 'dataset_dir' 'labelmap'
  # example: python convert_pascal_2_ssd.py ~/datasets/baitcam ~/datasets/baitcam/baitcam_labelmap.prototxt --resize-width 300 --resize-height 300
  ```

3. This should create:

  ```Shell  
  $baitcam/ssd_ImageSets        # Contains train.txt, val.txt, val_name_size.txt imagesets (SSD format)
  $baitcam/train_lmdb           # Contains .mdb files
  $baitcam/val_lmdb             # Contains .mdb files
  ```
4. Convert to YOLOv2 data format:

  ```Shell
  cd $ROOT/data/data_utils
  python convert_pascal_2_yolov2.py -h     # To see arguments
  python convert_pascal_2_yolov2.py 'dataset_dir' 'labelmap'
  # example: python convert_pascal_2_ssd.py ~/datasets/baitcam ~/datasets/baitcam/baitcam_labelmap.prototxt
  ```

5. This should create:

  ```Shell
  $baitcam/yolov2_ImageSets          # Contains train.txt and val.txt imagesets (YOLO format)
  $baitcam/yolov2_Annotations        # Contains .txt annotations (YOLO format)
  ```

6. Create custom anchor boxes for dataset (optional):

  ```Shell
  cd $ROOT/data/data_utils
  python k_means_anchor_boxes.py -h     # To see arguments
  python k_means_anchor_boxes.py 'dataset_dir'
  # example: python convert_pascal_2_ssd.py ~/datasets/baitcam --k 5 --info
  ```

7. This should create:

  ```Shell
  $baitcam/custom_anchor_boxes/5_anchor_boxes.txt
  ```

### Train a model:
1. Download pretrained weights, link in $ROOT/imagenet_models/download.txt
2. Run unified detection system:

  ```Shell
  cd $ROOT
  python run.py -h     # To see arguments
  python run.py --dataset 'dataset_name' --method 'method_name' --model 'model_name'
  # example: python run.py --dataset baitcam --method ssd --model VGG16_reduced
  ```

3. This should create output directory:

  ```Shell
  $ROOT/output/baitcam/ssd/VGG16_reduced    # Contains model weights, logfile, config etc.
  ```


### Evaluate a model:
1. Run unified detection system:

  ```Shell
  cd $ROOT
  python run.py -h     # To see arguments
  python run.py --eval --output_dir 'output_dir'
  # example: python run.py --eval --output_dir /output/baitcam/ssd/VGG16_reduced
  ```
2. This should create results directory:

  ```Shell
  $ROOT/output/baitcam/ssd/VGG16_reduced/results    # Contains result files and Precision-recall curve
  ```

### Detect with a model:
1. Run unified detection system:

  ```Shell
  cd $ROOT
  python run.py -h     # To see arguments
  python run.py --detect --output_dir 'output_dir' --image_dir 'image_dir'
  # example: python run.py --detect --output_dir /output/baitcam/ssd/VGG16_reduced --image_dir ~/baitcam_test_images
  ```

2.  This should create detections file:

  ```Shell
  ~/baitcam_test_images_detections.txt
  ```

### Visualize detection results:
1. Run visualization script:

  ```Shell
  cd $ROOT
  python visualize_detections.py -h     # To see arguments
  python visualize_detections.py 'detection_file'
  # example: python visualize_detections.py ~/baitcam_test_images_detections.txt --labelmap ~/datasets/baitcam/baitcam_labelmap.prototxt --vis_thresh 0.5
  ```
