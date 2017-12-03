# Install
$ROOT refers to the unified-detection-system root directory.

### Installing 'caffe-rcnn-ssd':
1. Dependencies: CUDA, cuDNN, OpenCV, lmdb, leveldb, BLAS, Boost, protobuf, glog, gflags, hdf5

  ```Shell
  cd $ROOT/frameworks/caffe-rcnn-ssd
  cp Makefile.config.default Makefile.config # Modify Makefile.config to your Caffe installation (Makefile.config.baitcam was used in thesis)
  mkdir build
  cd build
  cmake ..
  make -j8
  
  # Add Caffe to $CAFFE_ROOT and $PYTHONPATH environment variables (should also be added to ~/.bashrc):
  export CAFFE_ROOT=$ROOT/frameworks/caffe-rcnn-ssd
  export PYTHONPATH=$ROOT/frameworks/caffe-rcnn-ssd/python:$PYTHONPATH
  ```

### Installing 'py-faster-rcnn':
1. Dependencies: CUDA, OpenCV, Numpy, Cython, EasyDict, PyYAML and Matplotlib.

  ```Shell
  cd $ROOT/frameworks/py-faster-rcnn/lib
  make
  ```


### Installing 'darknet':
1. Dependencies: CUDA, cuDNN and OpenCV.

  ```Shell
  cd $ROOT/frameworks/darknet
  make darknet-cpp
  ```
