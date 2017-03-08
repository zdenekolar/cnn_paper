# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:16:22 2017

@author: User
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
number_samples = 50000

def augment(input_dir, output_dir, prefix, target_size=(300,300)):
    '''
    Augments data from input directory and saves it in output directory.
    :param input_dir: string input directory
    :param output_dir: string output directory
    :param prefix: string prefix for saved files
    :param target_size: tuple size of the images
    :return:
    '''
    datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='reflect',
            )

    i = 0
    for batch in datagen.flow_from_directory(
            input_dir,
            batch_size=1,
            target_size=target_size,
            class_mode='binary',
            save_to_dir=output_dir,
            save_prefix=prefix,
            save_format='jpeg',
            shuffle=True):

        i += 1
        if i > number_samples:
            break  # otherwise the generator would loop indefinitely


if __name__ == '__main__':

    input_dir = r'D:\ConvNet\cnn_paper\data\augmentation1000'
    output_dir = r'D:\ConvNet\cnn_paper\data\augmented_slightly'
    prefix = 'guardrail'

    augment(input_dir, output_dir, prefix)
