from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from shutil import copyfile
import numpy as np
import glob

def testing_stas(model, pos_data_dir, neg_data_dir):
    '''
    Test the network and return stats.
    :param model:
    :param test_x:
    :param test_y:
    :return:
    '''

    img_width = 300
    img_height = 300

    pos_imgs = glob.glob(pos_data_dir + '/*.*')
    neg_imgs = glob.glob(neg_data_dir + '/*.*')
    print(len(pos_imgs))
    print(len(neg_imgs))

    samples = pos_imgs + neg_imgs
    train_labels = np.array([1] * len(pos_imgs) + [0] * len(neg_imgs))

    predicted_labels = np.zeros(len(samples))

    for i in range(len(pos_imgs) + len(neg_imgs)):
        print(i)
        img = load_image(samples[i], img_width, img_height)
        predicted_labels[i] = model.predict_classes(img, batch_size=1)[0]
        # print(predicted_labels[i] == train_labels[i])
        if predicted_labels[i] == 1 and train_labels[i] == 0:
            copyfile(samples[i], r'fp\{}.jpg'.format(i))
        elif predicted_labels[i] == 0 and train_labels[i] == 1:
            copyfile(samples[i], r'fn\{}.jpg'.format(i))

    # Post-processing
    accuracy = accuracy_score(train_labels, predicted_labels)
    cm = confusion_matrix(train_labels, predicted_labels)
    f1 = f1_score(train_labels, predicted_labels)
    prec = precision_score(train_labels, predicted_labels)
    rec = recall_score(train_labels, predicted_labels)
    clas_rep = classification_report(train_labels, predicted_labels)
    print(accuracy)
    print(cm)
    print(f1)
    print(prec)
    print(rec)
    print(clas_rep)
    return accuracy

def load_image(path, img_width, img_height):
    img = load_img(path).resize((300, 300))
    x = img_to_array(img)/255
    x = x.reshape((1,) + (3, 300, 300))
    return x
