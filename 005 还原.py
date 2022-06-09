import gc
gc.enable()

####################
# 获取所有
####################

import os
parameter = 'left'

path = '004 flat\\' \
       +parameter
files = os.listdir(path)
# print(files)

import cv2
import numpy as np

####################
# 图像显示函数
####################
'''
def imshow(string='imshow', name=None):
    cv2.imshow(string, name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
####################
# 长方形变圆环
####################

def rec2cir(img):
    
    h,w,_ = img.shape
    
    radius = w / (np.pi*2)
    cir_img = np.zeros((int(2*radius)+1, int(2*radius)+1, 3), dtype='uint8')
    circle_center = ((int(2*radius)+1)//2, (int(2*radius)+1)//2)
    
    for row in np.arange(0, img.shape[0], 0.5):
        for col in np.arange(0, img.shape[1], 0.5):
            
            rho = radius - row - 1
            theta = (col+1)*(np.pi*2) / img.shape[1] # + origin_theta
            p_x = int(circle_center[0] + rho*np.sin(theta) + 0.5)
            p_y = int(circle_center[1] - rho*np.cos(theta) - 0.5)
            cir_img[p_y, p_x, :] = img[int(row), int(col), :]

    # 采用插值方案
    # 模糊不太行
    # cir_img = cv2.blur(cir_img, (3, 3))
    
    return cir_img

####################
# 函数
####################

def process(file):

    if file != 'Thumbs.db':
        print(file, end=' ')

        img = cv2.imread(path+'\\'+file)
        rec2cir_img = rec2cir(img)
        # imshow('cir2rec_img', cir2rec_img)

        # 保存
        cv2.imwrite('005 back\\'+parameter+'\\'+file, rec2cir_img)

####################
# 多进程
####################

from multiprocessing import Pool

if __name__ == '__main__':
    
    pool = Pool() # python来决定 processes=16/24/32 没区别，python不会用爆cache
    pool.map(process, files)
    pool.close()
    pool.join()
