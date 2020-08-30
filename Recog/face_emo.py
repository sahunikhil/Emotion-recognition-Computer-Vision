import tensorflow as tf
import tensorflow.keras as keras
from keras.models import *
from keras.models import Sequential
import cv2
import numpy as np

def predict(im):
    img_width, img_height = 224, 224
    input_shape = (img_width, img_height, 3)
    model = Sequential() 
    model = load_model('expface.h5')
    img = cv2.imread(im)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    classes = np.argmax(model.predict(img), axis=-1)
    #print(classes[0])
    a=classes
    if(a[0]==0):
        return('Angry')
    elif(a[0]==1):
        return('Happy')
    elif(a[0]==2):
        return('Surprise')
    elif(a[0]==3):
        return('Disgust')
    elif(a[0]==4):
        return('Sad')
    elif(a[0]==5):
        return('Fear')

