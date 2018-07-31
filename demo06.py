#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''doc'''

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
biases = tf.Variable(tf.zeros([3]))
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)