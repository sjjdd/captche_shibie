import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cnn_utils

# char_num=62
# Y_channel=4
# m=5000
# IMAGE_PATH = './train/'
# train_label_path = './train_label.csv'
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



c_tensor=tf.constant([[[1,2,3,2],[4,5,6,2]],[[7,8,9,2],[10,11,12,2]],[[7,8,9,2],[10,11,12,2]]])
print(c_tensor.get_shape().as_list()[0])