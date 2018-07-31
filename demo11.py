#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''doc'''

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

sess = tf.InteractiveSession()

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.clip_by_value(v, 2.5, 4.5).eval())

v = tf.constant([[1.0, 2.0, 3.0]])

print(tf.log(v).eval())

v1 = tf.constant([[1.0,2.0],[3.0,4.0]])
v2 = tf.constant([[5.0,6.0],[7.0,8.0]])
print((v1 * v2).eval())

print(tf.matmul(v1, v2).eval())

print(tf.reduce_mean(v1).eval())

v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])
sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())

print(tf.where(tf.greater(v1,v2),v1,v2).eval())