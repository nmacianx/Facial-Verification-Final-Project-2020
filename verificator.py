from keras import backend as K
import time
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

from keras.models import model_from_json

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

# print('Se compilara el modelo...')
# FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
# print('Modelo compilado exitosamente.\n')
# print('Se cargaran los pesos de la red...')
# start = time.time()
# load_weights_from_FaceNet(FRmodel)
# end = time.time()
# print('Pesos cargados.')
# print('Tiempo de carga: {}'.format(end-start))

# print('Se guardara el modelo')
# # serialize model to JSON
# model_json = FRmodel.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# FRmodel.save_weights("model.h5")
# print("Saved model to disk")

print('Iniciando carga...')
start = time.time()
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
FRmodel = model_from_json(loaded_model_json)
# load weights into new model
FRmodel.load_weights("model.h5")
print("Loaded model from disk")
print('Se compilara el modelo...')
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print('Modelo compilado exitosamente.\n')
end = time.time()
print('Tiempo de carga: {}'.format(end-start))


# Aca iria toda la preparacion de la base de datos de imagenes, #
# en facenet.py lo hace de una forma distinta
# Hardcodeado, le pongo a mano las identidades a cargar
database = {}
database["NicoMacian"] = img_path_to_encoding("images/nicomacian.png", FRmodel)
database["NicoMacian1"] = img_path_to_encoding("images/evaluacion.png", FRmodel)

def verify(image_path, identity):
    encoding = img_to_encoding(image_path,FRmodel)
    dist = np.linalg.norm(np.subtract(database[identity],encoding))
    return dist