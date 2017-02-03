# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:24:56 2017

@author: Zdenek
"""

import os
import random
from shutil import copyfile


def randomize(path_from, path_to, number) :
    


for c in range(count):
    file = random.choice(os.listdir(path_from))
    print(c)
    copyfile(path_from + file, path_to + str(c) + '.jpg')
    
    
    
if __name__ == '__main__':
    
    
    path_from = r'D:\ConvNet\cnn_paper\data\train\1/'
    path_to = r'D:\ConvNet\cnn_paper\data\random1000/1/'

    number = 2048

    randomize(path_from, path_to, count)