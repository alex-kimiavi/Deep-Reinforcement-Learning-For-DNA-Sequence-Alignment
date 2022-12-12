from __future__ import division

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import nn

from collections import namedtuple
import matplotlib.pyplot as plt
import scipy.misc
import os
#matplotlib inline

BUFFER_SIZE = 50000


class C51():
    def __init__(self, h_size, env, name, LEARNING_RATE, n_step, n_atoms):
        
        self.v_max = 10
        self.v_min = -10
        self.delta_z = (self.v_max - self.v_min)/(float(n_atoms - 1))
        self.z = [self.v_min + i*self.delta_z for i in range(n_atoms)]
        WINDOW_SIZE = env.win_size
        CONV_FILTER_SIZE_X = [3,3,3,3]
        CONV_FILTER_SIZE_Y = [3,3,3,3]
        CONV_STRIDE_X = [3,1,1,3]
        CONV_STRIDE_Y = [3,1,1,3]
        CONV_LAYER_NUM = 4
        CONV_FILTER_NUM = [8,32,32,64]
        IMAGE_SIZE = [2*(WINDOW_SIZE+2),8,3]
        self.scalarInput = tf.placeholder(shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
        depthwise_filter1 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[0], CONV_FILTER_SIZE_Y[0], 3, 1), name=name+"_depthwise_filter1")
        pointwise_filter1 = tf.get_variable(shape=[1, 1, 3, CONV_FILTER_NUM[0]], name=name+"_pointwise_filter1")
        self.conv1 = nn.separable_conv2d(self.imageIn, depthwise_filter1, pointwise_filter1, strides=[1,CONV_STRIDE_X[0],CONV_STRIDE_Y[0],1], padding='SAME')
        self.relu1 = nn.leaky_relu(self.conv1, name=name+"_relu1")
        depthwise_filter2 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[1], CONV_FILTER_SIZE_Y[1], CONV_FILTER_NUM[0], 1), name=name+"_depthwise_filter2")
        pointwise_filter2 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[0], CONV_FILTER_NUM[1]], name=name+"_pointwise_filter2")
        self.conv2 = nn.separable_conv2d(self.relu1, depthwise_filter2, pointwise_filter2, strides=[1,CONV_STRIDE_X[1],CONV_STRIDE_Y[1],1], padding='SAME')
        self.relu2 = nn.leaky_relu(self.conv2, name=name+"_relu2")
        depthwise_filter3 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[2], CONV_FILTER_SIZE_Y[2], CONV_FILTER_NUM[1], 1), name=name+"_depthwise_filter3")
        pointwise_filter3 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[1], CONV_FILTER_NUM[2]], name=name+"_pointwise_filter3")
        self.conv3 = nn.separable_conv2d(self.relu2, depthwise_filter3, pointwise_filter3, strides=[1,CONV_STRIDE_X[2],CONV_STRIDE_Y[2],1], padding='SAME')
        self.relu3 = nn.leaky_relu(self.conv3, name=name+"_relu3")
        self.maxpool1 = nn.max_pool(self.relu3, ksize=[1,3,1,1], strides=[1,3,1,1], padding='VALID')
        if np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3) >= 2:
            conv_filter4 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[3], CONV_FILTER_SIZE_Y[3], CONV_FILTER_NUM[2], CONV_FILTER_NUM[3]), name=name+"_conv_filter4")
            self.conv4 = nn.conv2d(self.maxpool1, conv_filter4, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='SAME')
            self.relu4 = nn.leaky_relu(self.conv4, name=name+"_relu4")
            self.maxpool2 = nn.max_pool(self.relu4, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
            LAST_CONV_FILTER = [np.floor(np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3)/2),1]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[3], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool2, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
        else:
            LAST_CONV_FILTER = [np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3),3]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[2], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool1, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
        self.relu5 = nn.leaky_relu(self.conv5, name=name+"_relu5")
        self.flat = slim.flatten(self.relu5)
        # self.d1 = slim.fully_connected(self.flat, 512, activation_fn=tf.nn.leaky_relu)
        self.a0 = slim.fully_connected(self.flat, n_atoms, activation_fn=tf.nn.softmax)
        self.a1 = slim.fully_connected(self.flat, n_atoms, activation_fn=tf.nn.softmax)
        self.a2 = slim.fully_connected(self.flat, n_atoms, activation_fn=tf.nn.softmax)
        self.raw_out = tf.convert_to_tensor([self.a0, self.a1, self.a2])


        self.q = tf.reduce_sum(tf.multiply(tf.concat([self.a0, self.a1, self.a2], axis=0), self.z), axis=1)
        self.dbg = tf.print(self.q)
        self.predict = tf.argmax(self.q, name=name + '_action_out')
        self.m_probs = tf.placeholder(shape=[3, None, 51], dtype=tf.float32, name=name + '_probs_in')
        # self.loss = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor([self.a0, self.a1, self.a2]), self.m_probs )

        self.loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(tf.convert_to_tensor([self.a0, self.a1, self.a2])), self.m_probs), axis=0)))

        self.trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.update_model = self.trainer.minimize(self.loss)

class Qnetwork():
    def __init__(self, h_size, env, LEARNING_RATE, n_step):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        WINDOW_SIZE = env.win_size
        CONV_FILTER_SIZE_X = [9,6,3]
        CONV_FILTER_SIZE_Y = [9,6,3]
        CONV_STRIDE_X = [3,3,1]
        CONV_STRIDE_Y = [3,3,1]
        CONV_LAYER_NUM = 4
        CONV_FILTER_NUM = [32,64,64]
        IMAGE_SIZE = [3*(WINDOW_SIZE+2),12,3]
        LAST_CONV_FILTER = [np.ceil((WINDOW_SIZE+2)/3),2]
        self.scalarInput = tf.placeholder(shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=CONV_FILTER_NUM[0], kernel_size=[CONV_FILTER_SIZE_X[0], CONV_FILTER_SIZE_Y[0]], stride=[CONV_STRIDE_X[0], CONV_STRIDE_Y[0]], padding='SAME',
            biases_initializer=None)
        print(np.shape(self.conv1))
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=CONV_FILTER_NUM[1], kernel_size=[CONV_FILTER_SIZE_X[1], CONV_FILTER_SIZE_Y[1]], stride=[CONV_STRIDE_X[1], CONV_STRIDE_Y[1]], padding='SAME',
            biases_initializer=None)
        print(np.shape(self.conv2))
        self.conv3 = slim.conv2d( \
            inputs=self.conv2, num_outputs=CONV_FILTER_NUM[2], kernel_size=[CONV_FILTER_SIZE_X[2], CONV_FILTER_SIZE_Y[2]], stride=[CONV_STRIDE_X[2], CONV_STRIDE_Y[2]], padding='SAME',
            biases_initializer=None)
        print(np.shape(self.conv3))
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=h_size, kernel_size=[LAST_CONV_FILTER[0], LAST_CONV_FILTER[1]], stride=[1,1], padding='VALID',
            biases_initializer=None)
        #depthwise_filter4 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER
        print(np.shape(self.conv4))

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()

        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        print(self.conv4)
        print(self.streamA)
        print(self.AW)

        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # For various exploration strategies
        self.Temp = tf.placeholder(shape=[None],dtype=tf.float32)
        self.Qdist = slim.softmax(self.Qout/self.Temp)
        self.keep_per = tf.placeholder(shape=[None],dtype=tf.float32)
        self.conv4 = slim.dropout(self.conv4,self.keep_per[0])

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.updateModel = self.trainer.minimize(self.loss)

class SSDnetwork():
    def __init__(self, h_size, env, name, LEARNING_RATE, n_step):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        WINDOW_SIZE = env.win_size
        CONV_FILTER_SIZE_X = [3,3,3,3]
        CONV_FILTER_SIZE_Y = [3,3,3,3]
        CONV_STRIDE_X = [3,1,1,3]
        CONV_STRIDE_Y = [3,1,1,3]
        CONV_LAYER_NUM = 4
        CONV_FILTER_NUM = [8,32,32,64]
        IMAGE_SIZE = [2*(WINDOW_SIZE+2),8,3]
        self.scalarInput = tf.placeholder(shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
        depthwise_filter1 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[0], CONV_FILTER_SIZE_Y[0], 3, 1), name=name+"_depthwise_filter1")
        pointwise_filter1 = tf.get_variable(shape=[1, 1, 3, CONV_FILTER_NUM[0]], name=name+"_pointwise_filter1")
        self.conv1 = nn.separable_conv2d(self.imageIn, depthwise_filter1, pointwise_filter1, strides=[1,CONV_STRIDE_X[0],CONV_STRIDE_Y[0],1], padding='SAME')
        print(np.shape(self.conv1))
        self.relu1 = nn.relu(self.conv1, name=name+"_relu1")
        print(np.shape(self.relu1))
        depthwise_filter2 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[1], CONV_FILTER_SIZE_Y[1], CONV_FILTER_NUM[0], 1), name=name+"_depthwise_filter2")
        pointwise_filter2 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[0], CONV_FILTER_NUM[1]], name=name+"_pointwise_filter2")
        self.conv2 = nn.separable_conv2d(self.relu1, depthwise_filter2, pointwise_filter2, strides=[1,CONV_STRIDE_X[1],CONV_STRIDE_Y[1],1], padding='SAME')
        print(np.shape(self.conv2))
        self.relu2 = nn.relu(self.conv2, name=name+"_relu2")
        print(np.shape(self.relu2))
        depthwise_filter3 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[2], CONV_FILTER_SIZE_Y[2], CONV_FILTER_NUM[1], 1), name=name+"_depthwise_filter3")
        pointwise_filter3 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[1], CONV_FILTER_NUM[2]], name=name+"_pointwise_filter3")
        self.conv3 = nn.separable_conv2d(self.relu2, depthwise_filter3, pointwise_filter3, strides=[1,CONV_STRIDE_X[2],CONV_STRIDE_Y[2],1], padding='SAME')
        print(np.shape(self.conv3))
        self.relu3 = nn.relu(self.conv3, name=name+"_relu3")
        print(np.shape(self.relu3))
        self.maxpool1 = nn.max_pool(self.relu3, ksize=[1,3,1,1], strides=[1,3,1,1], padding='VALID')
        print(np.shape(self.maxpool1))
        if np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3) >= 2:
            conv_filter4 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[3], CONV_FILTER_SIZE_Y[3], CONV_FILTER_NUM[2], CONV_FILTER_NUM[3]), name=name+"_conv_filter4")
            self.conv4 = nn.conv2d(self.maxpool1, conv_filter4, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='SAME')
            print(np.shape(self.conv4))
            self.relu4 = nn.relu(self.conv4, name=name+"_relu4")
            print(np.shape(self.relu4))
            self.maxpool2 = nn.max_pool(self.relu4, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
            LAST_CONV_FILTER = [np.floor(np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3)/2),1]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[3], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool2, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
            print(np.shape(self.maxpool2))
        else:
            LAST_CONV_FILTER = [np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3),3]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[2], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool1, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
        print(np.shape(self.conv5))
        self.relu5 = nn.relu(self.conv5, name=name+"_relu5")
        print(np.shape(self.relu5))

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.relu5, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        print(self.conv5)
        print(self.streamA)
        print(self.AW)
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.updateModel = self.trainer.minimize(self.loss)

class DiffSSDnetwork():
    def __init__(self, h_size, env, name, LEARNING_RATE, n_step):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        WINDOW_SIZE = env.win_size
        CONV_FILTER_SIZE_X = [3,3,3,3]
        CONV_FILTER_SIZE_Y = [3,3,3,3]
        CONV_STRIDE_X = [3,1,1,3]
        CONV_STRIDE_Y = [3,1,1,3]
        CONV_LAYER_NUM = 4
        CONV_FILTER_NUM = [16,32,32,64]
        IMAGE_SIZE = [2*(WINDOW_SIZE+2),8,8]
        self.scalarInput = tf.placeholder(shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
        depthwise_filter1 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[0], CONV_FILTER_SIZE_Y[0], 8, 1), name=name+"_depthwise_filter1")
        pointwise_filter1 = tf.get_variable(shape=[1, 1, 8, CONV_FILTER_NUM[0]], name=name+"_pointwise_filter1")
        self.conv1 = nn.separable_conv2d(self.imageIn, depthwise_filter1, pointwise_filter1, strides=[1,CONV_STRIDE_X[0],CONV_STRIDE_Y[0],1], padding='SAME')
        print(np.shape(self.conv1))
        self.relu1 = nn.relu(self.conv1, name=name+"_relu1")
        print(np.shape(self.relu1))
        depthwise_filter2 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[1], CONV_FILTER_SIZE_Y[1], CONV_FILTER_NUM[0], 1), name=name+"_depthwise_filter2")
        pointwise_filter2 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[0], CONV_FILTER_NUM[1]], name=name+"_pointwise_filter2")
        self.conv2 = nn.separable_conv2d(self.relu1, depthwise_filter2, pointwise_filter2, strides=[1,CONV_STRIDE_X[1],CONV_STRIDE_Y[1],1], padding='SAME')
        print(np.shape(self.conv2))
        self.relu2 = nn.relu(self.conv2, name=name+"_relu2")
        print(np.shape(self.relu2))
        depthwise_filter3 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[2], CONV_FILTER_SIZE_Y[2], CONV_FILTER_NUM[1], 1), name=name+"_depthwise_filter3")

        pointwise_filter3 = tf.get_variable(shape=[1, 1, CONV_FILTER_NUM[1], CONV_FILTER_NUM[2]], name=name+"_pointwise_filter3")
        self.conv3 = nn.separable_conv2d(self.relu2, depthwise_filter3, pointwise_filter3, strides=[1,CONV_STRIDE_X[2],CONV_STRIDE_Y[2],1], padding='SAME')
        print(np.shape(self.conv3))
        self.relu3 = nn.relu(self.conv3, name=name+"_relu3")
        print(np.shape(self.relu3))
        self.maxpool1 = nn.max_pool(self.relu3, ksize=[1,3,1,1], strides=[1,3,1,1], padding='VALID')
        print(np.shape(self.maxpool1))
        if np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3) >= 2:
            conv_filter4 = tf.get_variable(shape=(CONV_FILTER_SIZE_X[3], CONV_FILTER_SIZE_Y[3], CONV_FILTER_NUM[2], CONV_FILTER_NUM[3]), name=name+"_conv_filter4")
            self.conv4 = nn.conv2d(self.maxpool1, conv_filter4, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='SAME')
            print(np.shape(self.conv4))
            self.relu4 = nn.relu(self.conv4, name=name+"_relu4")
            print(np.shape(self.relu4))
            self.maxpool2 = nn.max_pool(self.relu4, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
            LAST_CONV_FILTER = [np.floor(np.ceil(np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3)/3)/2),1]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[3], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool2, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
            print(np.shape(self.maxpool2))
        else:
            LAST_CONV_FILTER = [np.floor(np.ceil(2*(WINDOW_SIZE+2)/3)/3),3]
            conv_filter5 = tf.get_variable(shape=(LAST_CONV_FILTER[0], LAST_CONV_FILTER[1], CONV_FILTER_NUM[2], h_size), name=name+"_conv_filter5")
            self.conv5 = nn.conv2d(self.maxpool1, conv_filter5, strides=[1,CONV_STRIDE_X[3],CONV_STRIDE_Y[3],1], padding='VALID')
        print(np.shape(self.conv5))
        self.relu5 = nn.relu(self.conv5, name=name+"_relu5")
        print(np.shape(self.relu5))

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.relu5, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        print(self.conv5)
        print(self.streamA)
        print(self.AW)
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.updateModel = self.trainer.minimize(self.loss)

class FFTDQNnetwork():
    def __init__(self, h_size, env, LEARNING_RATE, n_step):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        FREQ_SIZE = env.freq_size
        IMAGE_SIZE = [FREQ_SIZE,4,4]
        self.scalarInput = tf.placeholder(shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*IMAGE_SIZE[2]], dtype=tf.float32)
        self.fc1 = slim.fully_connected(inputs=self.scalarInput, num_outputs=128, activation_fn=tf.nn.relu)
        self.fc2 = slim.fully_connected(inputs=self.fc1, num_outputs=128, activation_fn=tf.nn.relu)
        self.fc3 = slim.fully_connected(inputs=self.fc2, num_outputs=128, activation_fn=tf.nn.relu)
        self.fc4 = slim.fully_connected(inputs=self.fc3, num_outputs=h_size, activation_fn=tf.nn.relu)

        print(np.shape(self.fc4))

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.fc4, 2, 1)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()

        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        print(self.fc4)
        print(self.streamA)
        print(self.AW)

        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.updateModel = self.trainer.minimize(self.loss)


class SumTree(object):
    data_pointer = 0

    def __init__(self, buffer_size=BUFFER_SIZE):

        self.capacity = buffer_size
        self.tree = np.zeros(2*self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, tree_index, priority):
        
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1)//2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):

        parent_index = 0
        while True:
            left_child_index = 2*parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
        

class per_experience_buffer():
    
    PER_e = 0.01
    PER_a = 0.8
    PER_b = 0.5

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.

    def __init__(self, capacity=BUFFER_SIZE):
        self.tree = SumTree(buffer_size=capacity)

    def add(self, experience):

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)
    
    def sample(self, n):
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)
        priority_segment = self.tree.total_priority/n

        for i in range(n):
            a, b = priority_segment*i, priority_segment*(i+1)
            value = np.random.uniform(a,b)
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i] = index
            minibatch.append([data[0][j] for j in range(5)])
        return b_idx, minibatch
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class experience_buffer():
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states,states.size)

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def copyGraphOp(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(var.value()))
    return op_holder

def copyGraphOp2(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(var.value()))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def copyGraph(op_holder,sess):
    for op in op_holder:
        sess.run(op)
