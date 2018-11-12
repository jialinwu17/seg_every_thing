#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from tqdm import tqdm
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import cPickle
from caffe2.python import workspace
#print(sys.path)
#sys.path.remove('/mnt/storage/jialinwu/detectron/detectron')
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--use-vg3k',
        dest='use_vg3k',
        help='use Visual Genome 3k classes (instead of COCO 80 classes)',
        action='store_true'
    )
    parser.add_argument(
        '--thresh',
        default=0.7,
        type=float,
        help='score threshold for predictions',
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = (
        dummy_datasets.get_vg3k_dataset()
        if args.use_vg3k else dummy_datasets.get_coco_dataset())


    if args.im_or_folder == 'train0':
        #im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
        im_list = cPickle.load(open('train_list.pkl'))
        pre_folder = '/mnt/storage/jialinwu/detectron/detectron/datasets/data/coco/coco_train2014'
        im_list = im_list[:40000]
    elif args.im_or_folder == 'train1':
        im_list = cPickle.load(open('train_list.pkl'))
        pre_folder = '/mnt/storage/jialinwu/detectron/detectron/datasets/data/coco/coco_train2014'
        im_list = im_list[40000:]
    elif args.im_or_folder == 'val':
        im_list = cPickle.load(open('val_list.pkl'))
        pre_folder = '/mnt/storage/jialinwu/detectron/detectron/datasets/data/coco/coco_val2014'
    elif args.im_or_folder == 'test0':
        im_list = cPickle.load(open('test_list.pkl'))
        im_list = im_list[:40000]
        pre_folder = '../ARNet/image_captioning/data/images/test2015'
    elif args.im_or_folder == 'test1':
        im_list = cPickle.load(open('test_list.pkl'))
        im_list = im_list[40000:]
        pre_folder = '../ARNet/image_captioning/data/images/test2015'
    for i in tqdm(range(len(im_list))): 
        im_name = pre_folder + '/' + im_list[i]
        if im_name[-4:] != '.jpg':
            continue
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        imgid = int(im_list[i][:-4].split('_')[-1])
        if args.im_or_folder == 'val':
            save_name = '/mnt/storage/jialinwu/seg_every_thing/npz_features/coco_val2014/%d.npz'%imgid
        else:
            save_name = '/mnt/storage/jialinwu/seg_every_thing/npz_features/coco_train2014/%d.npz'%imgid
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, save_name, timers=timers
            )
        '''
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=2
        )
        '''

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
