from utils.load_data import load_database
from utils.inception_network import *

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
database = load_database(FRmodel)