import pickle
import os
import cv2 as cv
import numpy as np
from os.path import isfile

from proyecto.utils.verification.fr_utils import img_path_to_encoding, img_to_encoding
import proyecto.data.settings as SETTINGS

def load_database(model, data_dir, new=False):
    file_name = os.path.join(data_dir, SETTINGS.db_name)
    if isfile(file_name) and not new:
        print('Reading dataset files...')
        with open(file_name, 'rb') as f:
	        face_encodings = pickle.load(f)
    else:
        face_encodings = create_database(model, data_dir, file_name)
    return face_encodings
    
def create_database(model, data_dir, file_name):
    print('Creating dataset files...')
    face_encodings = {}
    for file in os.listdir(os.path.join(data_dir,'img')):
        if file.endswith('.png'):
            name = os.path.splitext(file)[0]
            face_encodings[name] = img_path_to_encoding(os.path.join(os.path.join(data_dir,'img'), file), model)
    with open(file_name, 'wb') as f:
        pickle.dump(face_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Dataset files created')
    return face_encodings

def update_database(database, model, image, identity, data_dir):
    database[identity] = img_to_encoding(image, model)  # Create embedding for the new face
    image_to_save = cv.resize(image, (96, 96)) 
    image_path = identity + '.png'
    image_path = os.path.join(os.path.join(data_dir, 'img'), image_path)
    cv.imwrite(image_path, image_to_save.astype(np.uint8))
    file_name = os.path.join(data_dir, SETTINGS.db_name)
    with open(file_name, 'wb') as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Dataset file updated')