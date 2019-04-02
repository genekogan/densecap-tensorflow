# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
# Train a dense captioning model
# Code adapted from faster R-CNN project
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join as pjoin
import sys
import six
import glob
import argparse
import json
import numpy as np
import tensorflow as tf
from lib.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
from lib.dense_cap.train import get_training_roidb, train_net
from lib.dense_cap.test import im_detect, sentence
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
import pprint
from lib.fast_rcnn.nms_wrapper import nms
from runway import RunwayModel


densecap = RunwayModel()


@densecap.setup
def setup(alpha=0.5):
    global net, vocab
    ckpt_dir = 'output/ckpt/'
    vocabulary = 'output/ckpt/vocabulary.txt'

    # load network
    net = resnetv1(num_layers=50) # vgg16() resnetv1(num_layers=50, 101, 152)
    net.create_architecture("TEST", num_classes=1, tag='pre')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    # load vocab
    vocab = ['<PAD>', '<SOS>', '<EOS>']
    with open(vocabulary, 'r') as f:
        for line in f:
            vocab.append(line.strip())

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    saver = tf.train.Saver()
    sess = tf.Session(config=tfconfig)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restored from {}'.format(ckpt.model_checkpoint_path))

    return sess


@densecap.command('caption', inputs={'image': 'image'}, outputs={'captions': 'vector', 'scores': 'vector', 'boxes': 'vector'})
def caption(sess, inp):
    img = np.array(inp['image'])
    scores, boxes, captions = im_detect(sess, net, img, None, use_box_at=-1)
    pos_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(pos_dets, cfg.TEST.NMS)
    pos_dets = pos_dets[keep, :]
    pos_scores = scores[keep]
    pos_captions = [sentence(vocab, captions[idx]) for idx in keep]
    pos_boxes = boxes[keep, :]
    return dict(captions=np.array(pos_captions), scores=np.array(pos_scores), boxes=np.array(pos_boxes))


if __name__ == '__main__':
    densecap.run()
