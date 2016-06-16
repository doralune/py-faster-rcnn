#!/usr/bin/env python
# coding: utf-8

"""
Demo script showing detections in a sample image
using Faster R-CNN

edit lib/fast_rcnn/config.py
__C.TEST.SCALES = (128,)
__C.TEST.MAX_SIZE = 350
"""

import pdb
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from nms import py_cpu_nms as pcn
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import shutil
import gc
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def get_color(args, zs):
    max_z = 1.0
    min_z = 0.0
    range_z = max_z - min_z
    cs = []
    cmap = args.cmap
    cmap_step = args.cmap_step
    if range_z > 0.0:
        for z in zs:
            i = int((z - min_z) / range_z * (cmap_step - 1))
            cs.append(cmap(i))
    else:
        cs = [cmap(0)] * pos_map_num
    return cs

def vis_detections(args, ax, iax, im, class_names, dets, thresh=0.5):
    im = im[:, :, (2, 1, 0)]
    #ax.imshow(im, aspect='equal')
    iax.set_data(im)
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    color = get_color(args, dets[:, -1])

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        col = color[i]
        class_name = class_names[i]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          #edgecolor='red', linewidth=3.5)
                          edgecolor=col, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1], 
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, pad=5.0),
                va='top',
                fontsize=14, color='white')

    ax.set_title(('detections with '
                  'p(class | box) >= {:.1f}').format(thresh),
                  fontsize=14)

def detect(net, im_file, thresh=0.3):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    vclss = []
    vdets = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, thresh)
        dets = dets[keep, :]
        vclss += [cls] * len(dets)
        vdets += list(dets)
    vdets = np.array(vdets)
    return im, vclss, vdets

def demo(args, net):
    #CONF_THRESH = 0.5
    #NMS_THRESH = 0.3
    fig = plt.figure(figsize=(8,6), dpi=(args.display_width / 8))
    ax = plt.subplot2grid((1,1), (0,0))
    font_size = 'x-large'
    plt.ion()
    plt.show()

    shutil.copy(args.image_file, args.image_copy_file)
    img = cv2.imread(args.image_copy_file)
    img = img[:, :, (2, 1, 0)]
    iax = ax.imshow(img, aspect='equal')
    ax.axis('off')
    plt.draw()

    while True:
        gc.collect()

        # copy image before processing
        try:
            shutil.copy(args.image_file, args.image_copy_file)
            img, vclss, vdets = detect(net, args.image_copy_file, thresh=args.NMS_THRESH)
        except:
            continue

        # plot
        #plt.cla()
        #plt.axis([0, 1, 0, 1])
        ax.texts[:] = []
        ax.patches[:] = []
        vis_detections(args, ax, iax, img, vclss, vdets, thresh=args.CONF_THRESH)
        #ax.axis('off')
        plt.draw()
        #pdb.set_trace()
        #plt.pause(0)
        #time.sleep(0.1)
        if args.out_image_file:
            #plt.tight_layout()
            plt.savefig(args.out_image_file)
            print('Wrote %s' % args.out_image_file)
        #plt.clf()
        #ax = plt.gca()
        #pdb.set_trace()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    # image
    parser.add_argument("--image-file", dest="image_file", default='../../module_caffe/cam/dummy.jpg')
    parser.add_argument("--image-copy-file", dest="image_copy_file", default='tools/input.jpg')
    #parser.add_argument("--out-image-file", dest="out_image_file", default='tools/output.jpg')
    parser.add_argument("--out-image-file", dest="out_image_file", default=None)
    # color
    parser.add_argument("--cmap-name", dest="cmap_name", default='jet')
    parser.add_argument("--cmap-step", dest="cmap_step", type=int, default=2**16)

    args = parser.parse_args()
    args.cmap = plt.get_cmap(args.cmap_name, args.cmap_step)

    raw = raw_input('Insert display_width(480px): ')
    try:
        raw = int(raw)
    except:
        raw = 480
    args.display_width = raw

    raw = raw_input('Insert CONF_THRESH(0.5): ')
    try:
        raw = int(raw)
    except:
        raw = 0.5
    args.CONF_THRESH = raw

    raw = raw_input('Insert NMS_THRESH(0.3): ')
    try:
        raw = int(raw)
    except:
        raw = 0.3
    args.NMS_THRESH = raw

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    im_name = args.image_file
    demo(args, net)
