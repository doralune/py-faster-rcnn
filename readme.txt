[ Installation ]

https://github.com/rbgirshick/py-faster-rcnn

########################################################
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
########################################################
cd $FRCN_ROOT/lib
edit setup.py
delete line 125-142 for ignore .cu file
comment out gpu_nms in lib/fast_rcnn/nms_wrapper.py
force cpu in lib/fast_rcnn/nms_wrapper.py
make
########################################################
cd $FRCN_ROOT/caffe-fast-rcnn

cp Makefile.config.example Makefile.config
CPU_ONLY := 1
USE_OPENCV := 0
ANACONDA_HOME := $(HOME)/anaconda
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		$(ANACONDA_HOME)/include/python2.7 \
		$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
WITH_PYTHON_LAYER := 1

Fix Library Install name on OS X for @rpath Linking #3696
https://github.com/BVLC/caffe/pull/3696/files

make all
make pycaffe
########################################################
cd $FRCN_ROOT
./data/scripts/fetch_faster_rcnn_models.sh
########################################################
./tools/demo.py
########################################################

lib/fast_rcnn/config.py
__C.TEST.SCALES = (200,)
__C.TEST.MAX_SIZE = 500
