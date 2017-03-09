import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from matplotlib import pyplot as plt
from keras.regularizers import l1, l2
import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# path to the model weights file.
weights_path = r'D:\ConvNet\cnn_paper/vgg16_weights.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = r'data/random1000'
validation_data_dir = r'data/validation_A'
nb_train_samples = 4096
nb_validation_samples = 2944
nb_epoch = 100
dropout = 0.5
l2_c = 0
batch = 32

def load_data():
    '''
    Load features from keras_bottleneck_train.py
    Return as training and validation data
    :return:
    '''

    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
    
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))    
    
    return train_data, train_labels, validation_data, validation_labels


def train_svm(c, k, train_data, train_labels, validation_data, validation_labels):
    '''
    Train SVM on data from keras_bottleneck_train.py
    :param c:
    :param k:
    :param train_data:
    :param train_labels:
    :param validation_data:
    :param validation_labels:
    :return:
    '''

    cls = svm.SVC(C=c, kernel = k)
    
    cls.fit(train_data, train_labels)
    predictions = cls.predict(validation_data)
    
    score = accuracy_score(predictions, validation_labels)
    f1 = f1_score(predictions, validation_labels)
    precision = precision_score(predictions, validation_labels)
    recall = recall_score(predictions, validation_labels)
    print('C, accuracy, f1, precision, recall', c, score, f1, precision, recall)
    return score

    
if __name__ == '__main__':
    accs = []
    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    train_data, train_labels, validation_data, validation_labels = load_data()
    for k in ['linear', 'rbf']:
        for C in vals:
            acc = train_svm(C, k, train_data, train_labels, validation_data, validation_labels)
            accs.append(acc)
            
    print(accs)
