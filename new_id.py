from os import system, name
from os.path import isfile
import cv2 as cv
import sys

db_dir = 'images/'
# define our clear function 
def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


listo = False
comprobado = False
while not comprobado:
    while not listo:
        clear()
        print('Programa para agregar una nueva identidad al sistema.')
        print('=====================================================\n')
        nombre = input('Escriba el nombre de la persona que se cargara: ')
        print('Se cargara {}.'.format(nombre))
        confirma = input('\nEs correcto el nombre? S/N ')
        if confirma == 's' or confirma == 'S':
            listo = True
    nombre = nombre.lower().replace(" ", "_")
    if isfile(db_dir+nombre+'.png'):
        listo = False
        print('\n\nLo sentimos, esa identidad ya fue cargada en el sistema.')
        input('Presione una tecla para volver a comenzar.')
    else:
        comprobado = True
    
input('\n\nPresione una tecla para capturar la imagen.')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "cfg/yolov3-tiny.cfg";
# modelConfiguration = "cfg/yolov3-tiny-prn-modif.cfg";
modelWeights = "trained-weights/yolov3-tiny_last.weights";
# modelWeights = "trained-weights/yolov3-tiny-prn-modif_last.weights";
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
cap = cv.VideoCapture(0)
while True:        
    hasFrame, frame = cap.read()
    frame = frame[32:448, 112:528]
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    cv.imshow('Captura de nueva identidad', frame)
    cv.waitKey(1)
cap.release()
cv.destroyAllWindows()