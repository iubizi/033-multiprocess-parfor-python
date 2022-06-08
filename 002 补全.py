####################
# 获取所有
####################

import os
parameter = 'right'

path = 'CroppedIrises_cut\\CroppedEyes_'+parameter
files = os.listdir(path)
# print(files)

####################
# 图片加边框
####################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def process(file):
    
    if file != 'Thumbs.db': # 系统文件
        print(file, end=' ')
        
        img = cv2.imread(path+'\\'+file)
        # print(img.shape, end='\t')

        new_img = np.zeros([max(img.shape), max(img.shape), 3])
        # print(new_img.shape)

        x_edge = int((max(img.shape)-img.shape[0])/2)
        y_edge = int((max(img.shape)-img.shape[1])/2)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    new_img[i+x_edge, j+y_edge, k] = img[i, j, k]

        cv2.imwrite('data\\'+parameter+'\\'+file, new_img)

####################
# 多进程
####################

from multiprocessing import Pool

if __name__ == '__main__':
    
    pool = Pool() # python来决定 processes=16/24/32 没区别，python不会用爆cache
    pool.map(process, files)
    pool.close()
    pool.join()
