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
from model.faster_rcnn.resnet_w2v import resnet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2) + 1e-8)

name_vectors = {0:[float(x) for x in "-0.6742 0.85137 -0.44563 -0.13542 0.66321 0.40091 -0.17317 -0.31843 -0.50706 -0.16611 -0.0037329 -0.15644 0.49711 0.42111 -0.15725 -0.29234 -0.49116 -0.41583 0.40411 -0.98499 0.038447 0.58983 -0.097437 0.13938 0.21717 -0.79006 -1.1404 0.063862 0.091416 0.13035 2.6491 -0.39185 0.10433 -1.2945 0.25436 0.68762 -0.24406 -0.55442 -0.26576 -0.19875 0.4121 0.48016 0.10084 -0.0076807 0.14454 -0.2255 0.78955 0.49908 0.1241 0.06661".split()], 
                1:[float(x) for x in "0.27203 -0.83471 -0.77436 -0.52342 -0.7846 0.43548 -0.55723 -0.12461 0.40705 -0.26957 0.9665 0.65333 -0.16101 -0.12138 -1.2511 -0.3532 -0.99238 1.3659 -1.4554 -0.61021 0.2708 0.23935 -0.68575 -0.38805 0.31965 -0.43329 -0.56755 0.32242 0.7346 0.21644 0.022579 -0.34303 -0.19963 1.0519 0.80017 -0.27862 0.70645 0.18759 -0.46561 0.033631 0.9176 -1.2649 0.23333 0.1328 0.32079 0.038267 0.2414 0.0043794 0.046835 0.20238".split()],
                2:[float(x) for x in "-0.41716 0.16271 0.18208 0.040874 -0.76743 0.47832 -0.95683 -0.69713 1.3446 -0.59146 0.49005 -0.59504 -0.96514 0.41526 -0.053253 -0.65851 -0.33963 1.4676 -0.51452 -1.5919 0.44235 -0.10577 -1.4277 1.1811 0.2842 -0.92328 -0.2138 0.10129 0.61985 -0.21404 1.3418 0.53977 -0.65476 0.55278 0.38827 0.34769 0.37724 0.45527 -0.30526 0.97048 -0.016631 -0.36535 0.060563 0.33608 0.098575 -0.96056 0.31872 -1.1316 0.91703 -0.65762".split()],
                3:[float(x) for x in "0.78675 0.079368 -0.76597 0.1931 0.55014 0.26493 -0.75841 -0.8818 1.6468 -0.54381 0.33519 0.44399 1.089 0.27044 0.74471 0.2487 0.2491 -0.28966 -1.4556 0.35605 -1.1725 -0.49858 0.35345 -0.1418 0.71734 -1.1416 -0.038701 0.27515 -0.017704 -0.44013 1.9597 -0.064666 0.47177 -0.03 -0.31617 0.26984 0.56195 -0.27882 -0.36358 -0.21923 -0.75046 0.31817 0.29354 0.25109 1.6111 0.7134 -0.15243 -0.25362 0.26419 0.15875".split()],
                4:[float(x) for x in "0.96989 0.075746 0.157 -0.10599 -0.6147 0.085639 -1.3175 0.63774 0.84952 -1.0661 0.023076 0.69078 0.49954 0.89659 -0.11993 -0.68484 -0.055934 0.89771 -1.6127 -0.93314 0.34222 0.56598 -0.49005 0.064676 1.27 -1.5572 0.47292 0.36106 0.58891 -0.25283 2.1768 0.029947 -0.45634 1.1527 0.38669 0.26368 0.32151 -0.50073 -0.34753 0.094688 -0.10259 -0.34041 1.2957 0.019886 0.056735 -0.68682 -0.24096 -0.91495 0.73288 -0.95561".split()],
                5:[float(x) for x in "0.20635 0.2606 -0.094723 -0.73396 0.72598 0.5099 -0.39352 -0.45703 0.49335 1.3791 0.10285 0.14997 0.41506 -0.19039 1.0527 0.16514 -0.16717 0.8092 -0.97394 -1.753 0.34632 -0.053064 0.33046 -0.021036 -0.78655 -1.0088 -0.30341 1.6766 0.90808 -0.39309 1.2131 0.21588 -0.87778 1.3756 0.57432 0.35111 0.39926 0.33184 1.2035 -0.21218 1.2316 0.58557 -0.40531 0.37376 0.16584 0.56948 -0.13898 -0.29062 0.56082 -0.94112".split()],
                6:[float(x) for x in "0.84772 0.070253 0.96791 -0.27164 -0.37617 0.31978 -1.3108 0.091093 0.59919 -0.90217 -0.050876 -0.83886 -0.61596 0.29642 -0.42189 -0.21969 -0.94006 1.2221 -0.66526 -0.57745 0.76126 0.51459 -0.88565 1.5135 0.42326 -1.2947 0.45522 0.67073 0.80188 -0.65449 2.4117 0.62445 -0.046631 0.37524 1.0103 0.25259 1.0913 -0.79427 -0.17027 1.4866 -0.24077 0.021904 -0.01603 -0.44319 -0.13914 0.013311 -0.49432 -0.57696 1.1997 -0.39707".split()],
                7:[float(x) for x in "0.47685 -0.084552 1.4641 0.047017 0.14686 0.5082 -1.2228 -0.22607 0.19306 -0.29756 0.20599 -0.71284 -1.6288 0.17096 0.74797 -0.061943 -0.65766 1.3786 -0.68043 -1.7551 0.58319 0.25157 -1.2114 0.81343 0.094825 -1.6819 -0.64498 0.6322 1.1211 0.16112 2.5379 0.24852 -0.26816 0.32818 1.2916 0.23548 0.61465 -0.1344 -0.13237 0.27398 -0.11821 0.1354 0.074306 -0.61951 0.45472 -0.30318 -0.21883 -0.56054 1.1177 -0.36595".split()],
                8:[float(x) for x in "0.45281 -0.50108 -0.53714 -0.015697 0.22191 0.54602 -0.67301 -0.6891 0.63493 -0.19726 0.33685 0.7735 0.90094 0.38488 0.38367 0.2657 -0.08057 0.61089 -1.2894 -0.22313 -0.61578 0.21697 0.35614 0.44499 0.60885 -1.1633 -1.1579 0.36118 0.10466 -0.78325 1.4352 0.18629 -0.26112 0.83275 -0.23123 0.32481 0.14485 -0.44552 0.33497 -0.95946 -0.097479 0.48138 -0.43352 0.69455 0.91043 -0.28173 0.41637 -1.2609 0.71278 0.23782".split()],
                9:[float(x) for x in "-1.0443 0.49202 -0.75978 -0.39224 0.81217 -0.039287 0.016706 -0.68629 -0.078359 -1.3214 -0.15354 0.20438 -0.46503 1.2145 -0.18217 0.27451 -0.24086 0.71145 0.3247 -0.7132 0.66721 0.71307 -0.10394 -0.38439 -0.2026 -1.4419 0.42644 0.59436 -1.3615 0.0013784 1.8734 -0.11334 -0.88115 -0.21715 -0.56606 0.14152 0.27673 0.99962 1.0567 -0.29428 -0.3139 0.12729 -0.54363 0.39652 -0.32527 0.30536 0.15128 -1.0889 -0.20867 -0.052605".split()],
                10:[float(x) for x in "0.61253 -0.48167 -0.74199 -0.55203 -0.007596 1.6101 -0.88565 -0.81981 1.5144 -0.22804 0.55367 -0.18392 0.7049 -0.36931 1.0668 1.1077 0.19709 0.24731 -0.68395 0.5475 -0.038255 -0.78989 0.61131 0.31473 0.50215 -1.6535 -0.42782 1.0404 0.29429 -0.36889 1.3148 -0.18443 0.092753 0.77572 -0.54845 -0.14645 0.51128 0.047248 0.41781 -0.18324 -0.44197 -0.25237 -0.3359 0.3096 1.9192 0.3396 -0.27341 -0.01316 0.64974 -0.85857".split()],
                11:[float(x) for x in "0.08052    2.4489    -1.82518   -0.0477     2.00389   -1.69875 -0.751511  -0.50848   -0.428588  -1.31098   -1.68757   -0.32697  0.07998    1.65463    0.90282    0.26249    0.13324    0.62698  0.371472  -2.6305     1.60213    0.56334   -0.38732    1.69456 -0.58381   -0.632158  -0.41       1.51932   -0.09594    0.21045  4.788      0.99336   -0.44077    0.63905    0.6379     1.42181  0.47585    1.95477    0.327167  -0.12653    0.77951   -0.10258 -0.179286   2.03756    0.1777877  0.9459     0.39777   -0.778347  0.99945   -0.651".split()],
                12:[float(x) for x in "0.11008 -0.38781 -0.57615 -0.27714 0.70521 0.53994 -1.0786 -0.40146 1.1504 -0.5678 0.0038977 0.52878 0.64561 0.47262 0.48549 -0.18407 0.1801 0.91397 -1.1979 -0.5778 -0.37985 0.33606 0.772 0.75555 0.45506 -1.7671 -1.0503 0.42566 0.41893 -0.68327 1.5673 0.27685 -0.61708 0.64638 -0.076996 0.37118 0.1308 -0.45137 0.25398 -0.74392 -0.086199 0.24068 -0.64819 0.83549 1.2502 -0.51379 0.04224 -0.88118 0.7158 0.38519".split()],
                13:[float(x) for x in "-0.20454 0.23321 -0.59158 -0.29205 0.29391 0.31169 -0.94937 0.055974 1.0031 -1.0761 -0.0094648 0.18381 -0.048405 -0.35717 0.26004 -0.41028 0.51489 1.2009 -1.6136 -1.1003 -0.23455 -0.81654 -0.15103 0.37068 0.477 -1.7027 -1.2183 0.038898 0.23327 0.028245 1.6588 0.26703 -0.29938 0.99149 0.34263 0.15477 0.028372 0.56276 -0.62823 -0.67923 -0.163 -0.49922 -0.8599 0.85469 0.75059 -1.0399 -0.11033 -1.4237 0.65984 -0.3198".split()],
                14:[float(x) for x in "0.14362 -1.1402 0.39368 0.18135 -0.094088 0.67473 -0.52618 0.21466 0.62416 -0.17217 0.67109 -1.1389 -0.84819 0.085305 0.20975 -0.59836 -0.78554 1.21 -0.90412 -1.009 0.42731 0.39614 -1.0663 0.66758 0.54771 -0.93963 -0.31805 0.14893 0.4489 -0.1986 0.20147 0.47226 -0.31627 0.83248 0.84036 0.40339 0.24902 -0.034884 -0.11794 0.89527 -0.33927 0.13761 -0.037933 -0.26963 0.85965 -1.174 0.31216 -0.62433 1.4447 -1.0968".split()],
                15:[float(x) for x in "0.61734 0.40035 0.067786 -0.34263 2.0647 0.60844 0.32558 0.3869 0.36906 0.16553 0.0065053 -0.075674 0.57099 0.17314 1.0142 -0.49581 -0.38152 0.49255 -0.16737 -0.33948 -0.44405 0.77543 0.20935 0.6007 0.86649 -1.8923 -0.37901 -0.28044 0.64214 -0.23549 2.9358 -0.086004 -0.14327 -0.50161 0.25291 -0.065446 0.60768 0.13984 0.018135 -0.34877 0.039985 0.07943 0.39318 1.0562 -0.23624 -0.4194 -0.35332 -0.15234 0.62158 0.79257".split()],
                16:[float(x) for x in "1.87558   0.70306  -0.13368   1.0749    0.344     0.359    -1.29974 -0.81254   0.38347   0.20391   0.19335  -0.2915    1.75308  -1.23573  1.04908   0.20029   0.01158   2.52851  -0.17232  -1.70577  -0.50728 -2.48107  -0.60158  -1.52139   0.59061  -0.46542   1.71796   1.25714  1.44733  -0.97309   2.35745  -1.07286  -0.67348  -0.21713   0.02439  0.835637 -0.481641  2.17821   1.30031   0.16752  -0.37618  -1.94368  0.213935 -0.00917   2.51475   1.63038  -0.311639 -1.106158 -0.94201 -0.58308".split()],
                17:[float(x) for x in "0.39026 -1.1357 -0.47646 -1.0185 0.44893 1.4499 -1.5251 -0.97275 1.5137 -0.58559 0.68279 -0.63901 1.6961 -1.233 0.50333 0.6791 1.0007 0.53626 -1.1749 -0.054361 -0.42404 -0.49371 0.43183 0.40727 0.74916 -0.53304 -0.33745 0.11388 0.47025 -0.47616 1.3084 0.1534 0.09754 1.0811 -0.20525 0.14341 -0.080796 0.17984 0.12433 0.12709 -0.31428 0.035969 -0.4609 0.53133 1.4986 -0.22609 -0.46637 -0.54343 -0.27696 -0.75601".split()],
                18:[float(x) for x in "-0.12335 0.23509 -0.7438 -0.50219 0.97023 -0.12771 0.039666 -0.67514 -0.2298 -1.2603 -0.18912 0.019617 -0.096285 1.3191 0.31878 1.3884 -0.41529 0.33527 0.94093 -0.50369 0.37852 1.104 -0.41287 -0.10135 -0.59901 0.060218 -0.21657 1.5896 0.57137 0.061694 0.66554 0.39239 -0.73251 1.401 -0.095607 0.66528 0.36132 -0.014117 0.068288 -0.62261 -0.023754 -0.087556 0.1543 0.5651 -1.0085 0.065175 0.2977 -1.0227 -0.21718 -0.49374".split()],
                19:[float(x) for x in "0.94971 0.34328 0.84504 -0.88519 -0.72078 -0.29309 -0.74678 0.65122 0.47295 -0.74011 0.1877 -0.38279 -0.55899 0.42952 -0.26984 -0.42383 -0.31236 1.3423 -0.78567 -0.6302 0.91819 0.21126 -0.57442 1.4549 0.75456 -1.6165 -0.0085015 0.0029134 0.51304 -0.47447 2.5306 0.85944 -0.30667 0.057765 0.66231 0.20804 0.64237 -0.5246 -0.053416 1.1404 -0.13703 -0.18361 0.45459 -0.50963 -0.025539 -0.02861 0.18048 -0.4483 0.40525 -0.36821".split()],
                20:[float(x) for x in "0.25626    -0.17277     1.52866     1.06081    -0.56051    -1.009539 -0.845252   -1.18025     1.38689    -0.24081     0.622749    0.121095  0.02118     2.1764      0.712973    1.2033     -0.87334     0.5934 -0.11       -0.34415     1.34936     1.39815     1.1527      1.63682  0.0212764  -2.23525    -0.977023    0.09278    -0.01013    -0.06802  5.4253      0.57827    -0.6189     -1.3413     -1.44589     0.63279  0.43006    -1.66446    -0.96338    -0.87314     0.99411     1.098152 -0.24369    -0.4259232  -0.697979   -0.101647   -0.04148     0.58366 -0.26210906  0.947361".split()]}

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
      args.imdbval_name = "voc_2007_test"
      #args.imdbval_name = "voc_2007_trainval+voc_2012_trainval" # to run eval on train set
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

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  
  #map_location={'cuda:0':'cuda:{}'.format(int(args.gpus[0]))}
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()
    #fasterRCNN.to(torch.device("cuda:{}".format(args.gpus[0])))

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_{}_{}_{}'.format(args.checksession, args.checkepoch,args.gpus[0])
  num_images = len(imdb.image_index)
  #num_images = len(roidb) # to run eval on train set
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      
      softmax = nn.Softmax(dim=1)
      res_vector = []
      for ii in range(scores.shape[0]): # for all 300 the bounding boxes of an image
          res_vector.append([])
          #res_vector[ii].append(-1)
          for k in range(0, 21): # we calculate the dot product with each possible class
              res_vector[ii].append(cosine_similarity(scores.cpu().numpy()[ii], np.array(name_vectors[k])))
              
    
      #scores = softmax(torch.tensor(res_vector).cuda())
      scores = torch.tensor(res_vector).cuda()
      
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
