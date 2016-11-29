### Basic Installation of faster-rcnn (for demo)
- reference from https://github.com/rbgirshick/py-faster-rcnn

- Clone the Faster R-CNN repository
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git $FRCN_ROOT

- Build the Cython modules
cd $FRCN_ROOT/lib
make

- Build Caffe and pycaffe
cd $FRCN_ROOT/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
  WITH_PYTHON_LAYER := 1 # must
  USE_CUDNN := 1 # recommended
make all
make pycaffe

- Build Caffe and pycaffe 2
cd $FRCN_ROOT/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
  WITH_PYTHON_LAYER := 1 # must
  USE_CUDNN := 1 # recommended
mkdir build
cd build
cmake ..
ccmake . # edit configuration if needed
  CUDNN_INCLUDE                    /usr/local/cuda-7.5/include
  CUDNN_LIBRARY                    /usr/local/cuda-7.5/lib64/libcudnn.so
make all
  Found cuDNN: ver. 3.0.07 found (include: /usr/local/cuda-7.5/include, library: /usr/local/cuda-7.5/lib64/libcudnn.so)

- Check pycaffe
cd $FRCN_ROOT/caffe-fast-rcnn/python
python
import caffe

- Download pre-computed Faster R-CNN detectors
cd $FRCN_ROOT
./data/scripts/fetch_faster_rcnn_models.sh
  saved at data/faster_rcnn_models.tgz
  extracted at data/faster_rcnn_models
  See data/README.md for details. These models were trained on VOC 2007 trainval.
    VGG16 comes from the Caffe Model Zoo
    ZF was trained at MSRA (smaller than VGG16)

- Run demo
cd $FRCN_ROOT
./tools/demo.py

### In case of cpu only
- Build the Cython modules
cd $FRCN_ROOT/lib
temporalily delete line 125-142 in lib/setup.py 
comment out gpu_nms in lib/fast_rcnn/nms_wrapper.py
force cpu in lib/fast_rcnn/nms_wrapper.py
make

- Build Caffe and pycaffe
CPU_ONLY := 1
(USE_OPENCV := 0)
ANACONDA_HOME := $(HOME)/anaconda
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		$(ANACONDA_HOME)/include/python2.7 \
		$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
PYTHON_LIB := $(ANACONDA_HOME)/lib
WITH_PYTHON_LAYER := 1

### Problem of OS X
Fix Library Install name on OS X for @rpath Linking #3696
https://github.com/BVLC/caffe/pull/3696/files

### Using small image configuration
lib/fast_rcnn/config.py
__C.TEST.SCALES = (200,)
__C.TEST.MAX_SIZE = 500
