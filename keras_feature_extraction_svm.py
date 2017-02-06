'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
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


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)#127.5-127.5)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu'))
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    print("finished loading VGGNet")
       
    f.close()

    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    print('Bottleneck features saved.')

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
    
    print('Bottleneck validation features saved.')

def load_data():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
    
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))    
    
    return train_data, train_labels, validation_data, validation_labels
    
def train_svm(c, train_data, train_labels, validation_data, validation_labels):
    

    cls = svm.SVC(C=c, kernel = 'rbf')
    
    cls.fit(train_data, train_labels)
    predictions = cls.predict(validation_data)
    
    score = accuracy_score(predictions, validation_labels)
    f1 = f1_score(predictions, validation_labels)
    precision = precision_score(predictions, validation_labels)
    recall = recall_score(predictions, validation_labels)
    print(c, score, f1, precision, recall)
    return score
    
    
    
if __name__ == '__main__':
#    save_bottleneck_features()
    accs = []
    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    a, b, c, d = load_data()
    for C in vals:
        acc = train_svm(C, a, b, c, d)
        accs.append(accs)
        
    print(accs)