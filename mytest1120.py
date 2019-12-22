# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow as tf2
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from PIL import Image
# import os
# import cnn_utils
# import random
# import time
# import math
# from tensorflow.python.framework import ops
#
# char_num=62
# Y_channel=4
# m=5000
# IMAGE_PATH = './train/'
# train_label_path = './train_label.csv'
# CAPTCHA_IMAGE_WIDHT=120
# CAPTCHA_IMAGE_HEIGHT=40
# #一张验证码需要识别的字符数以及每个字符可能的种类数
# CAPTCHA_LEN=4
# CHAR_SET_LEN=62
# #存放训练好的模型的路径
# MODEL_SAVE_PATH = './models/'
#
# # 构建卷积神经网络并训练
# def train_data_with_CNN(X_train,Y_train,X_test,Y_test,minibatch_size):
#     seed=1
#     # 初始化权值
#     def weight_variable(shape, name='weig;ht'):
#         init = tf.truncated_normal(shape, stddev=0.1)
#         var = tf.Variable(initial_value=init, name=name)
#         return var
#
#         # 初始化偏置
#     def bias_variable(shape, name='bias'):
#         init = tf.constant(0.1, shape=shape)
#         var = tf.Variable(init, name=name)
#         return var
#
#         # 卷积
#     def conv2d(x, W, name='conv2d'):
#         return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
#         # 池化
#
#     def max_pool_2X2(x, name='maxpool'):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#
#     def max_pool_3X3(x, name='maxpool'):
#         return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#         # 输入层
#
#     # 请注意 X 的 name，在测试model时会用到它
#     # X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name='data-input')
#     # Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
#     # x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1], name='x-input')
#     #自改X为三层通道的训练数据
#     x_input=tf.placeholder(tf.float32,[None,CAPTCHA_IMAGE_HEIGHT,CAPTCHA_IMAGE_WIDHT,3],name='x-input')
#     Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
#     # dropout,防止过拟合
#     # 请注意 keep_prob 的 name，在测试model时会用到它
#     keep_prob = tf.placeholder(tf.float32, name='keep-prob')
#     # # 第一层卷积
#     # W_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
#     # B_conv1 = bias_variable([32], 'B_conv1')
#     # conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
#     # conv1 = max_pool_2X2(conv1, 'conv1-pool')
#     # conv1 = tf.nn.dropout(conv1, keep_prob)
#     # # 第二层卷积
#     # W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
#     # B_conv2 = bias_variable([64], 'B_conv2')
#     # conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
#     # conv2 = max_pool_2X2(conv2, 'conv2-pool')
#     # conv2 = tf.nn.dropout(conv2, keep_prob)
#     # # 第三层卷积
#     # W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
#     # B_conv3 = bias_variable([64], 'B_conv3')
#     # conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
#     # conv3 = max_pool_2X2(conv3, 'conv3-pool')
#     # conv3 = tf.nn.dropout(conv3, keep_prob)
#
#     ###############自改####################
#     #第一层卷积+池化
#     W_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
#     B_conv1 = bias_variable([32], 'B_conv1')
#     conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
#     conv1 = max_pool_3X3(conv1, 'conv1-pool')
#     conv1 = tf.nn.dropout(conv1, keep_prob)
#     # 第二层卷积+池化
#     W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
#     B_conv2 = bias_variable([64], 'B_conv2')
#     conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
#     conv2 = max_pool_3X3(conv2, 'conv2-pool')
#     conv2 = tf.nn.dropout(conv2, keep_prob)
#     #第三层卷积
#     W_conv3 = weight_variable([5, 5, 64, 128], 'W_conv3')
#     B_conv3 = bias_variable([128], 'B_conv3')
#     conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
#     conv3 = tf.nn.dropout(conv3, keep_prob)
#     #第四层卷积
#     W_conv4 = weight_variable([3, 3, 128, 256], 'W_conv4')
#     B_conv4 = bias_variable([256], 'B_conv4')
#     conv4 = tf.nn.relu(conv2d(conv3, W_conv4, 'conv4') + B_conv4)
#     conv4 = tf.nn.dropout(conv4, keep_prob)
#     #第五层卷积+池化
#     W_conv5 = weight_variable([3, 3, 256, 512], 'W_conv5')
#     B_conv5 = bias_variable([512], 'B_conv5')
#     conv5 = tf.nn.relu(conv2d(conv4, W_conv5, 'conv5') + B_conv5)
#
#     conv5 = max_pool_3X3(conv5,'conv5-pool')
#     conv5 = tf.nn.dropout(conv5, keep_prob)
#     #第一层全连接
#     W_fc = weight_variable([5 * 15 * 512, 1024], 'W_fc')
#     B_fc = bias_variable([1], 'B_fc')
#
#     fc = tf.reshape(conv5, [-1, W_fc.get_shape().as_list()[0]])
#     fc = tf.nn.relu(tf.add(tf.matmul(fc, W_fc), B_fc))
#     fc = tf.nn.dropout(fc, keep_prob)
#     #第二层全连接
#     W_fc1 = weight_variable([1024, 2048], 'W_fc1')
#     B_fc1 = bias_variable([1], 'B_fc1')
#
#     fc1 = tf.nn.relu(tf.add(tf.matmul(fc, W_fc1), B_fc1))
#     fc1 = tf.nn.dropout(fc1, keep_prob)
#
#
#     # 全链接层
#     # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
#     # W_fc1 = weight_variable([5 * 15 * 64, 1024], 'W_fc1')
#     # B_fc1 = bias_variable([1024], 'B_fc1')
#
#     # #fc1 = tf.reshape(conv3, [-1, 20 * 8 * 64])
#     # fc1 = tf.reshape(conv5, [-1, W_fc1.get_shape().as_list()[0]])
#     # fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
#     # fc1 = tf.nn.dropout(fc1, keep_prob)
#
#     # 输出层
#     W_fc2 = weight_variable([2048, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
#     B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
#     output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')
#
#     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
#     optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
#
#     predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
#     labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')
#     # 预测结果
#     # 请注意 predict_max_idx 的 name，在测试model时会用到它
#     predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
#     labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
#     predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
#     accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
#
#     saver = tf.train.Saver()
#     config = tf.ConfigProto(allow_soft_placement=True,
#                                 log_device_placement=True)
#     config.gpu_options.per_process_gpu_memory_fraction = 0.6
#
#
#     with tf.Session(config=config) as sess:
#         sess.run(tf.global_variables_initializer())
#         steps = 0
#         for epoch in range(6000):
#             num_minibatches = int(m / minibatch_size)
#             seed = seed + 1
#             minibatches = cnn_utils.random_mini_batches(X_train, Y_train,num_minibatches,seed)
#             for minibatch in minibatches:
#                 (train_data, train_label)=minibatch
#                 op,pre = sess.run([optimizer,labels_max_idx], feed_dict={x_input: train_data, Y: train_label, keep_prob: 0.75})
#                 #print(pre)
#                 #print(pre.shape)
#
#             if steps % 1 == 0:
#                 acc = sess.run(accuracy, feed_dict={x_input: X_test, Y: Y_test, keep_prob: 1.0})
#                 print("steps=%d, accuracy=%f" % (steps, acc))
#                 if acc > 0.99:
#                     saver.save(sess, MODEL_SAVE_PATH + "crack_captcha.model", global_step=steps)
#                     break
#             steps += 1
#
#
#
#
# # 获取X的训练集
# X_train_orig = cnn_utils.get_X_train(IMAGE_PATH)
# Y_train_total = cnn_utils.get_Y_train(train_label_path, char_num, Y_channel, m)
# # 归一化数据
# X_train_total = X_train_orig / 255
# print('Y_train_total_shape: ', Y_train_total.shape)
# print('X_train_shape: ', X_train_total.shape)
# # 取百分之七十做训练集，百分之三十做交叉验证集
# X_train = X_train_total[0:3500, :, :, :]
# X_test = X_train_total[3500:5000, :, :, :]
# Y_train = Y_train_total[0:3500, :]
# Y_test = Y_train_total[3500:5000, :]
# print('X_train_shape: ', X_train.shape)
# print('Y_train_shape: ', Y_train.shape)
# print('X_test_shape: ', X_test.shape)
# print('Y_test_shape: ', Y_test.shape)
# train_data_with_CNN(X_train,Y_train,X_test,Y_test,64)
#
#
#
#
#
