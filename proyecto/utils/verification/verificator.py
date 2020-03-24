import time
import cv2
import os
from os.path import isfile
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
K.set_image_data_format('channels_first')

import proyecto.data.settings as SETTINGS
from proyecto.utils.verification.fr_utils import load_weights_from_FaceNet, img_to_encoding
from proyecto.utils.verification.inception_network import faceRecoModel
from proyecto.utils.verification.load_data import load_database

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

def verify(image_path, identity, model, database):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(np.subtract(database[identity], encoding))
    return dist

def initialize():
    if (isfile(os.path.join(SETTINGS.verif_model_dir,'model.json')) and isfile(os.path.join(SETTINGS.verif_model_dir,'model.h5'))):
        ## Load saved model
        print('Loading model from disk...')
        start = time.time()
        json_file = open(os.path.join(SETTINGS.verif_model_dir,'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        FRmodel = model_from_json(loaded_model_json)
        FRmodel.load_weights(os.path.join(SETTINGS.verif_model_dir,'model.h5'))
        FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        end = time.time()
        print('Model loaded successfully in {:.2f}s.\n'.format(end-start))
    else:
        ## Generate model from separate weights files
        print('No model on disk. Starting model from scratch...')
        start = time.time()
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        load_weights_from_FaceNet(FRmodel, SETTINGS.verif_weights_dir)
        end = time.time()
        print('Model loaded successfully in {:.2f}s.\n'.format(end-start))
        # Save generated model to reduce load times
        model_json = FRmodel.to_json()
        with open(os.path.join(SETTINGS.verif_model_dir,'model.json'), "w") as json_file:
            json_file.write(model_json)
        FRmodel.save_weights(os.path.join(SETTINGS.verif_model_dir,'model.h5'))
        print("Model has been saved.")
    database = load_database(FRmodel, SETTINGS.data_dir)
    return FRmodel, database