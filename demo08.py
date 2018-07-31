#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''doc'''

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, name="w1"))

w2 = tf.Variable(tf.random_normal([2, 2], stddev=1, name="w2"))

tf.assign(w1, w2, validate_shape=False)