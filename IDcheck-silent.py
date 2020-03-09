import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from verificator import *
import time

# Initialize the parameters
confThreshold = 0.7  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "cfg/yolov3-tiny.cfg";
# modelConfiguration = "cfg/yolov3-tiny-prn-modif.cfg";
modelWeights = "trained-weights/yolov3-tiny_last.weights";
# modelWeights = "trained-weights/yolov3-tiny-prn-modif_last.weights";
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Variables usadas para la identificacion
MAX_ATTEMPTS = 10
VERIF_THRESHOLD = 0.5
verifying = False
attempts = 0
identity = "NicoMacian1"
state = None # Puede ser "verificado" o "rechazado"

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    global verifying, attempts, state
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:            
            scores = detection[5:]
            #Me parece que las proximas 2 lineas no tienen sentido en el caso que reconocemos solo 1 clase
            #Si no me equivoco en la interpretacion de los resultados
            classId = np.argmax(scores)            
            confidence = scores[classId]

            #no sabria cual es la diferencia entre detection[4] y detection[5], tienen valores parecidos
            #el 5 es la confianza, el 4 no se            
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)   #Detection 0 es el centro en x
                center_y = int(detection[1] * frameHeight)  #Detection 1 es el centro en y
                width = int(detection[2] * frameWidth)      #Detection 2 es el ancho
                height = int(detection[3] * frameHeight)    #Detection 3 es la altura
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)    
    if len(indices)==1:
        # print('--------------')
        i = indices[0][0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right=left+width
        bottom=top+height
        #Calculo el tamanio porcentual
        #ancho = width*100/frameWidth
        altura = height*100/frameHeight
        # print('Ancho: {}%'.format(ancho))
        # print('Altura: {}%'.format(altura))
        if altura > 30:
            # Debemos recortar la prediccion para verificarla
            padding = 30
            part_image = frame[max(0, top-padding):min(frameHeight, bottom+padding), max(0, left-padding):min(frameWidth, right+padding)]
            # Guardo el recorte para probar contra que se va a comparar
            # cv.imwrite("recorte.png", part_image.astype(np.uint8))
            if not verifying:
                attempts = 1
                verifying = True
            else:
                attempts += 1
            
            if not (attempts > MAX_ATTEMPTS):
                dist = verify(part_image, identity)
                # image = cv.resize(part_image, (96, 96)) 
                # cv.imwrite("evaluacion.png", image.astype(np.uint8))
                # dist = 0.3
                if not (dist < VERIF_THRESHOLD):  
                   print('Verificando...')
                else:
                    print('{} verificado'.format(identity))
                    state = 'verificado'
            else:
                # identidad rechazada
                print('Identidad no verificada')
                state = 'rechazado'
            
        else:
            verifying = False
            print('Acercate mas a la camara')
    elif len(indices)>1:
        verifying = False
        print('Mas de una cara detectada')
    else:
        verifying = False
        print('No se detecto una cara')                    

#Si solo usamos la cam directamente, y el bloque anterior esta comentado
cap = cv.VideoCapture(0)
while True:        
    # start = time.time()
    # start1 = start
    # get frame from the video
    hasFrame, frame = cap.read()
    # obtener 416x416 del medio
    frame = frame[32:448, 112:528]
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # end = time.time()
    # print("antes de procesar: {}".format(end-start))
    
    # Presiona espacio para reiniciar
    if not (state is None):
        input('Presione una tecla para continuar')
        state = None
        verifying = False

    if state is None:
        # start = time.time()
        postprocess(frame, outs)
        # end = time.time()
        # print("procesamiento: {}".format(end-start))

    # start = time.time()

    #El siguiente bloque muestra tiempo de inferencia.
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Tiempo de inferencia: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    print(label)

    # end = time.time()
    # print("despues de procesar: {}".format(end-start))
    # print('tiempo de vuelta: {}'.format(end-start1))
    # print('--------------------------------')
