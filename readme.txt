[ Installation ]
https://github.com/rbgirshick/py-faster-rcnn
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git

cd $FRCN_ROOT/lib
make

cd $FRCN_ROOT/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
make all
make pycaffe

[ cpu only ]
cd $FRCN_ROOT/lib
temporalily delete line 125-142 in lib/setup.py 
comment out gpu_nms in lib/fast_rcnn/nms_wrapper.py
force cpu in lib/fast_rcnn/nms_wrapper.py
make

[ caffe ]
CPU_ONLY := 1
(USE_OPENCV := 0)
ANACONDA_HOME := $(HOME)/anaconda
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		$(ANACONDA_HOME)/include/python2.7 \
		$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
PYTHON_LIB := $(ANACONDA_HOME)/lib
WITH_PYTHON_LAYER := 1

Fix Library Install name on OS X for @rpath Linking #3696
https://github.com/BVLC/caffe/pull/3696/files

[ demo ]
cd $FRCN_ROOT
./data/scripts/fetch_faster_rcnn_models.sh
./tools/demo.py

[ use small image ]
lib/fast_rcnn/config.py
__C.TEST.SCALES = (200,)
__C.TEST.MAX_SIZE = 500
