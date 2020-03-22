import os.path

dirname = os.path.dirname

## Settings for the project to be used in main functions

#   VARIABLE VALUES
#=======================================================================================
#Face detection threshold
recog_threshold = 0.7

#Face detection non max suppression threshold
nms_threshold = 0.4

#Input image width
inp_width = 416

#Input image height
inp_height = 416

#Maximum number of attempts to verify before stopping
max_attempts = 10

#ID verification threshold (Eucledian distance)
verif_threshold = 0.5

#Maximum percentage a detected face can be relative to the frame height
face_height_max_threshold = 50

#Minimum a detected face can be relative to the frame height
face_height_min_threshold = 25

#Number of pixels to add as padding to the detected face in order to verify the face
face_padding = 30

#   DIRECTORY PATHS
#=======================================================================================
#Data directory path
data_dir = os.path.join(dirname(dirname(__file__)), 'data')

#Model directory path
model_dir = os.path.join(dirname(dirname(__file__)), 'model')

#Verification model directory path
verif_model_dir = os.path.join(model_dir, 'verification')

#Recognition model directory path
recog_model_dir = os.path.join(model_dir, 'recognition')

#Weights directory path
weights_dir = os.path.join(dirname(dirname(__file__)), 'weights')

#Verification model weights directory path
verif_weights_dir = os.path.join(weights_dir, 'verification')

#Recognition model weights directory path
recog_weights_dir = os.path.join(weights_dir, 'recognition')

#Path to obj.names classes file
classes_file = os.path.join(os.path.join(model_dir, 'recognition'), 'obj.names')

#Cfg directory path
cfg_dir = os.path.join(dirname(dirname(__file__)), 'cfg')

#Name of the file where the embeddings database will be saved
db_name = 'dataset_faces.dat'

#   CONFIGURATION FILES AND WEIGHTS TO BE USED
#=======================================================================================
#Recognition model weights file path, two alternatives
recog_weights = os.path.join(recog_weights_dir, 'yolov3-tiny_last.weights')
# recog_weights = os.path.join(recog_weights_dir, 'yolov3-tiny-prn-modif_last.weights')

#Cfg file path, there are two alternative files
cfg_recog = os.path.join(cfg_dir, 'yolov3-tiny.cfg')
# cfg_recog = os.path.join(cfg_dir, 'yolov3-tiny-prn-modif.cfg')

#   OTHER VARIABLES
#=======================================================================================
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIN_NAME = 'Sistema de deteccion facial'