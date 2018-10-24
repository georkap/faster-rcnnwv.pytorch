# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from voc_eval import voc_eval

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _do_python_eval(_year, _devkit_path, _image_set, _classes, output_dir='output'):
#        annopath = os.path.join(
#            self._devkit_path,
#            'VOC' + self._year,
#            'Annotations',
#            '{:s}.xml')
        annopath = os.path.join(
                _devkit_path,
                'VOC' + _year,
                'Annotations') + '\\{:s}.xml'
        imagesetfile = os.path.join(
            _devkit_path,
            'VOC' + _year,
            'ImageSets',
            'Main',
            _image_set + '.txt')
        cachedir = os.path.join(_devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(_year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(_classes):
            if cls == '__background__':
                continue
            filename = _get_voc_results_file_template('trainval', _devkit_path, _year).format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

def _load_image_set_index(_year, _image_set):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        _devkit_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + _year)
        _data_path = os.path.join(_devkit_path, 'VOC' + _year)
        image_set_file = os.path.join(_data_path, 'ImageSets', 'Main',
                                      _image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

def _get_voc_results_file_template(_image_set, _devkit_path, _year):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = "comp4" + '_det_' + _image_set + '_{:s}.txt'
        filedir = os.path.join(_devkit_path, 'results', 'VOC' + _year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

def _write_voc_results_file(classes, image_index, _year, all_boxes):
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        print('Writing {} VOC results file'.format(cls))
        _devkit_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + _year)
        filename = _get_voc_results_file_template('trainval', _devkit_path, _year).format(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--gpus', nargs='+', type=int, default=None)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      #args.imdbval_name = "voc_2007_test"
      args.imdbval_name = "voc_2007_trainval+voc_2012_trainval" # to run eval on train set
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  start = time.time()
  save_name = 'faster_rcnn_{}_{}'.format(args.checksession, args.checkepoch)
  if args.gpus is not None:
      save_name += "_{}".format(args.gpus[0])
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)

  det_file = os.path.join(output_dir, 'detections.pkl')


  with open(det_file, 'rb') as f:
    all_boxes = pickle.load(f)

  _year = '2007'
  image_index = _load_image_set_index(_year, 'trainval')
  _devkit_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + _year)
  _write_voc_results_file(imdb._classes, image_index, _year, np.array(all_boxes)[:,:len(image_index)])
  _do_python_eval(_year, _devkit_path, 'trainval', imdb._classes, output_dir)
  
  _year = '2012'
  image_index = _load_image_set_index(_year, 'trainval')
  _devkit_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + _year)
  _write_voc_results_file(imdb._classes, image_index, _year, np.array(all_boxes)[:,-len(image_index):])
  _do_python_eval(_year, _devkit_path, 'trainval', imdb._classes, output_dir)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
