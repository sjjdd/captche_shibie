# #给图片去噪
# import cv2
# for i in range(5000):
#     # print(i)
#     img=cv2.imread('./train/'+str(i+1)+'.jpg')
#     # r=cv2.medianBlur(img,3)
#     b=cv2.bilateralFilter(img,50,50,200)
#     cv2.imwrite('./dealImg/'+str(i+1)+'.jpg',b)
#     # cv2.imshow('original',img)
#     # cv2.imshow('res',b)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
# print('finish！')


#创建随机元组
# import random
# import numpy as np
# a=[]
# for i in range(3):
#     b=random.randint(0,255)
#     a.append(b)
# print(tuple(a))

# import cv2
# img=cv2.imread('./test/2045.jpg')
# b=cv2.bilateralFilter(img,9,5,50)
# cv2.imwrite('./xxx.jpg',b)
# cv2.imshow('original',img)
# cv2.imshow('res',b)
# cv2.waitKey()
# cv2.destroyAllWindows()

#图片锐化
# !/usr/bin/env python
# _*_ coding:utf-8 _*_
# import cv2 as cv
# import numpy as np
#
#
# def custom_blur_demo(image):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#     dst = cv.filter2D(image, -1, kernel=kernel)
#     # #再进行去噪
#     # b=cv.bilateralFilter(dst,9,5,50)
#     cv.imshow("custom_blur_demo", dst)
#     cv.imwrite('./xxx.jpg',dst)
#
#
# src = cv.imread("./test/2274.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# custom_blur_demo(src)
# cv.waitKey(0)
# cv.destroyAllWindows()


