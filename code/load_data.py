import pickle
import os
from os.path import isfile
from fr_utils import img_path_to_encoding

dirname=os.path.dirname
data_dir = os.path.join(dirname(dirname(dirname(__file__))), 'data')

def load_database(model, new=False):
    file_name = os.path.join(data_dir,'dataset_faces.dat')
    if isfile(file_name) and not new:
        with open(file_name, 'rb') as f:
	        face_encodings = pickle.load(f)
    else:
        # tengo que leer todos los archivos .png, encodear y guardar
        face_encodings = {}
        for file in os.listdir(data_dir):
            if file.endswith(".png"):
                name = os.path.splitext(file)[0]
                face_encodings[name] = img_path_to_encoding(os.path.join(data_dir, file), model)
        with open(file_name, 'wb') as f:
            pickle.dump(face_encodings, f, protocol=pickle.HIGHEST_PROTOCOL)
    return face_encodings
    