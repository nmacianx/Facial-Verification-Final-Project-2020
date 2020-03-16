from keras import backend as K
import time
K.set_image_data_format('channels_first')
import cv2
import os
from os.path import isfile
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from fr_utils import *
from inception_network import *
from load_data import *

model_dir = os.path.join(dirname(dirname(__file__)), 'model')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

def verify(image_path, identity):
    encoding = img_to_encoding(image_path,FRmodel)
    dist = np.linalg.norm(np.subtract(database[identity],encoding))
    return dist


if (isfile(os.path.join(model_dir,'model.json')) and isfile(os.path.join(model_dir,'model.h5'))):
    ## Load saved model
    print('Loading model from disk...')
    start = time.time()
    # load json and create model
    json_file = open(os.path.join(model_dir,'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FRmodel = model_from_json(loaded_model_json)
    # load weights into new model
    FRmodel.load_weights(os.path.join(model_dir,'model.h5'))
    print("Loaded model from disk")
    print('Model will be compiled...')
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    print('Model compiled successfully.\n')
    end = time.time()
    print('Load time: {}'.format(end-start))
else:
    ## Generate model from separate weights files
    print('Starting model from scratch...')
    print('Model will be compiled...')
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    print('Model compiled successfully.\n')
    print('Loading network\'s weights...')
    start = time.time()
    load_weights_from_FaceNet(FRmodel)
    end = time.time()
    print("Loaded model from disk")
    print('Load time: {}'.format(end-start))

    # Save generated model to reduce load times

    print('Model will be saved')
    # serialize model to JSON
    model_json = FRmodel.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    FRmodel.save_weights("model.h5")
    print("Saved model to disk")

# Load faces' encodings
database = load_database(FRmodel)