#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''doc'''

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

print(a.graph is tf.get_default_graph())
