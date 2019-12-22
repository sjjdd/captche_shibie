#导入包
#注意以下2行是避免了使用tensorflow2版本
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cnn_utils
import cv2
import time
from PIL import Image
import numpy as np
import pandas as pd
#存储图片名字和标签的列表用于写入提交的csv文件
picname=[]
picTab=[]
CAPTCHA_IMAGE_WIDHT=120
CAPTCHA_IMAGE_HEIGHT=40
#一张验证码需要识别的字符数以及每个字符可能的种类数
CAPTCHA_LEN=4
CHAR_SET_LEN=62
TEST_Image_Path='./test/'

# 初始化权值
def weight_variable(shape, name='weight'):
    init = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial_value=init, name=name)
    return var

        # 初始化偏置
def bias_variable(shape, name='bias'):
    init = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial_value=init, name=name)
    return var

        # 卷积
def conv2d(x, W, name='conv2d'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
        # 池化

def max_pool_2X2(x, name='maxpool'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool_3X3(x, name='maxpool'):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def cnn_graph(x, keep_prob, size, captcha_list=cnn_utils.CAPTCHA_LIST, captcha_len=cnn_utils.CAPTCHA_LEN):
    # 自改X为三层通道的训练数据
    image_height, image_width = size
    #print('image_height, image_width ',image_height,image_width)
    # x_image=tf.reshape(x,shape=[-1,image_height,image_width,1])
    print("***********************")
    print(x.shape)
    # 自改X为三层通道的训练数据
    #x_input = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 3], name='x-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
    # dropout,防止过拟合
    # 请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = 1

    ###############自改####################
    ###############自改####################
    # 第一层卷积+池化
    W_conv1 = weight_variable([5, 5, 1, 16], 'W_conv1')
    B_conv1 = bias_variable([16], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_3X3(conv1, 'conv1-pool')
    # conv1 = tf.nn.dropout(conv1, keep_prob)
    # 第二层卷积+池化
    W_conv2 = weight_variable([5, 5, 16, 32], 'W_conv2')
    B_conv2 = bias_variable([32], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_3X3(conv2, 'conv2-pool')
    # conv2 = tf.nn.dropout(conv2, keep_prob)
    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 32, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    # conv3 = tf.nn.dropout(conv3, keep_prob)
    # 第四层卷积
    W_conv4 = weight_variable([3, 3, 64, 128], 'W_conv4')
    B_conv4 = bias_variable([128], 'B_conv4')
    conv4 = tf.nn.relu(conv2d(conv3, W_conv4, 'conv4') + B_conv4)
    # conv4 = tf.nn.dropout(conv4, keep_prob)
    # 第五层卷积+池化
    W_conv5 = weight_variable([3, 3, 128, 256], 'W_conv5')
    B_conv5 = bias_variable([256], 'B_conv5')
    conv5 = tf.nn.relu(conv2d(conv4, W_conv5, 'conv5') + B_conv5)

    conv5 = max_pool_3X3(conv5, 'conv5-pool')
    # conv5 = tf.nn.dropout(conv5, keep_prob)
    # 第一层全连接
    W_fc = weight_variable([5 * 15 * 256, 1024], 'W_fc')
    B_fc = bias_variable([1], 'B_fc')

    fc1 = tf.reshape(conv5, [-1, W_fc.get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc), B_fc))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # 第二层全连接
    # W_fc1 = weight_variable([1024, 4069], 'W_fc1')
    # B_fc1 = bias_variable([1], 'B_fc1')
    #
    # fc1 = tf.nn.relu(tf.add(tf.matmul(fc, W_fc1), B_fc1))
    # fc1 = tf.nn.dropout(fc1, keep_prob)

    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    # W_fc1 = weight_variable([5 * 15 * 64, 1024], 'W_fc1')
    # B_fc1 = bias_variable([1024], 'B_fc1')

    # #fc1 = tf.reshape(conv3, [-1, 20 * 8 * 64])
    # fc1 = tf.reshape(conv5, [-1, W_fc1.get_shape().as_list()[0]])
    # fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    # fc1 = tf.nn.dropout(fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    return output


def captcha2text(image_list,height=CAPTCHA_IMAGE_HEIGHT,width=CAPTCHA_IMAGE_WIDHT):
    '''
    验证吗图片转化为文本
    :param image_list:
    :param height:
    :param width:
    :return:
    '''

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, height,width,1])
        keep_prob=tf.placeholder(tf.float32)
        y_conv=cnn_graph(x,keep_prob,(height,width))
        #y_conv.numpy().tolist()
        saver=tf.train.Saver()
        # print('y_conv', y_conv.eval())
        # image_list=np.reshape(image_list,(1,4800,3))
        saver.restore(sess,tf.train.latest_checkpoint('./models/'))
        predict=tf.argmax(tf.reshape(y_conv,[-1,cnn_utils.CAPTCHA_LEN,len(cnn_utils.CAPTCHA_LIST)]),2)
        #predict = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        #获取验证码的数出结果
        vector_list=sess.run(predict,feed_dict={x:image_list,keep_prob:1})
        print(vector_list.shape)
        #print(sess.run(predict))
        vector_list=vector_list.tolist()
        text_list=[cnn_utils.vec2text(vector) for vector in vector_list]
        print(text_list)
        # tf.print('y_cpnv',y_conv)
        return text_list

if __name__ == '__main__':

    #读取待预测数据
    X_predict = cnn_utils.get_X_train('./test/')
    X_predict = X_predict/255
    X_predict=np.expand_dims(X_predict,-1)
    print("*************")
    print(X_predict.shape)
    tf.reset_default_graph()
    Y_predict = captcha2text(X_predict)
    i=1
    while(i<=len(Y_predict)):
        picname.append(str(i) + '.jpg')
        picTab.append(Y_predict[i-1])
        i=i+1
    dataframe = pd.DataFrame({'ID': picname, 'label': picTab})
    print(dataframe)
    dataframe.to_csv("./submission.csv", index=False, sep=',', mode='w')

