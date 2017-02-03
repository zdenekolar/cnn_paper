# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:14:46 2017

@author: Zdenek
"""

from keras_bottleneck import bn_training

samples =  (1024, 4096, 512)
dropout = (0, 1, 0.1)
val = 1024
epoch = 50

train_data_dir, validation_data_dir, nb_training_samples, 
                nb_validation_samples, nb_epoch, dropout

training_folder = r''
validation_folder = r''

for s in range(samples[0], samples[1], samples[2]):
    for d in range(dropout[0], dropout[1], dropout[2]):
        bn_training (training_folder, validation_folder, s, val, epoch, d)
        
    