import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
import time
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']
CAPTCHA_LIST = NUMBER+LOW_CASE+UP_CASE
CAPTCHA_LEN = 4         # 验证码长度

# CHAR_SET = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a'
#             'b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q'
#             'r','s','t','u','v','w','x','y','z'...])

#数据在独热矩阵中排序按照数字，小写字母，大写字母排
def name2label(one_label,char_num,Y_channel):
    label = np.zeros(char_num*Y_channel)
    for i, c in enumerate(one_label):
        if '0'<=c<='9':
            idx = i * char_num + ord(c) - ord('0')
            label[idx] = 1
        elif 'a'<=c<='z':
            idx = i * char_num + ord(c) - ord('a') + ord('9')-ord('0')+1
            label[idx] = 1
        else:
            idx = i * char_num + ord(c) - ord('A') + ord('z') - ord('a') + ord('9')-ord('0')+2
            label[idx] = 1
    return label

def label_merge(Y_train_orig,Y_train,char_num,Y_channel,m):
    for i in range(m):
        temp = name2label(Y_train_orig[i],char_num,Y_channel)
        Y_train[:,i]=temp.T
    return Y_train

def get_image_file_name(imgPath):
    fileName = []
    X_train = []
    total = 0
    temp=os.listdir(imgPath)
    #os.listdir读取文件是乱序的，需要进行排序以对应标签的值
    #filename有序存储了所有文件名字  total统计 读取的数目，两个参数备用
    temp.sort(key=lambda x:int(x[:-4]))
    for filePath in temp:
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        img=Image.open(imgPath+captcha_name)
        img = img.convert("L")
        X_train.append(np.array(img))
        total += 1
    # print(np.array(X_train))
    '''
    [[[162 170 149 ... 168 165 161]
  [158 159 160 ... 165  46 169]
  [158 180 153 ... 175 153 167]
  ...
  [191  36 157 ... 161 154 156]
    '''
    return np.array(X_train)


def get_X_train(imgpath):
    X_train = get_image_file_name(imgpath)
    return np.array(X_train)

def get_Y_train(train_label_path,char_num,Y_channel,m):
    # 将标签转化为独热矩阵
    reader_y = pd.read_csv(train_label_path, names=['label'])
    Y_train_orig = np.squeeze(reader_y.values.tolist()[1:5001])
    Y_temp = np.zeros([char_num * Y_channel, m])  # 248*5000的标签
    Y_train = label_merge(Y_train_orig, Y_temp, char_num, Y_channel, m).T
    return np.array(Y_train)

#取得验证码图片的数据以及它的标签----未使用，单通道备用方案
def get_data_and_label(Y_train, fileName, filePath):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten()/255
    image_label = Y_train[:,int(fileName[:-4])-1]
    return image_data, image_label

#获取数据块集_暂未使用
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    # permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，shuffled_X直接打乱原数组
    #这里的permutation是输出一个将[0到m-1]的排序数组打乱
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning  划分出的数据块数量
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        #从打乱后的X与Y中顺序取出一个个64大小的数据块
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        #在mini_batches中不断扩展数据块，之所以变成上面形式是为了方便索引
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    #操作最后一个不完整的数据块
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def vec2text(vec, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    """
    验证码向量转为文本
    :param vec:
    :param captcha_list:
    :param captcha_len:
    :return: 向量的字符串形式
    """
    vec_idx = vec
    text_list = [captcha_list[int(v)] for v in vec_idx]
    return ''.join(text_list)

def convert2gray(img):
    #转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten()
    return image_data

