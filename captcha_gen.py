import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
import pandas as pd


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_LIST = NUMBER+LOW_CASE+UP_CASE
print(CAPTCHA_LIST)
CAPTCHA_LEN = 4         # 验证码长度
CAPTCHA_HEIGHT = 40     # 验证码高度
CAPTCHA_WIDTH = 120     # 验证码宽度
picname=[]
picTab=[]

def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    """
    随机生成定长字符串
    :param char_set: 备选字符串列表
    :param captcha_size: 字符串长度
    :return: 字符串
    """
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


def gen_captcha_text_and_image(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT, save=None,fileName=None):
    """
    生成随机验证码
    :param width: 验证码图片宽度
    :param height: 验证码图片高度
    :param save: 是否保存（None）
    :return: 验证码字符串，验证码图像np数组
    """
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captcha_text = random_captcha_text()
    captcha = image.generate(captcha_text)
    # 保存
    if save:
        image.write(captcha_text, './train/' +str(fileName) + '.jpg')
        #先将图片名和标注分别写入两个list中
        picname.append(str(fileName)+'.jpg')
        picTab.append(str(captcha_text))
        # print(picTab)
    captcha_image = Image.open(captcha)
    # 转化为np数组
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    for i in range(5000):
        t, im = gen_captcha_text_and_image(save=True,fileName=(i+5001))
        print(t, im.shape)      # (40, 120, 3)
    #最后将图片名字和标注写入csv文件
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'ID': picname, 'label': picTab})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("./train_label.csv", index=False, sep=',', mode='a')

'''

#将数据追加到csv文件中，perfect!
import pandas as pd

#任意的多组列表
a = [1,2,3]
b = [4,5,6]

#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name':a,'b_name':b})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("./train_label.csv",index=False,sep=',',mode='a')
'''

