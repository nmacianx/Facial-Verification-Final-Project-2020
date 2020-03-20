import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time

import proyecto.data.settings as SETTINGS
from proyecto.utils.verification.verificator import verify, initialize

verification_model, database = initialize(SETTINGS.verif_model_dir, SETTINGS.data_dir, SETTINGS.verif_weights_dir) 

# Load names of classes
with open(SETTINGS.classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


# Give the configuration and weight files for the model and load the network using them.

net = cv.dnn.readNetFromDarknet(SETTINGS.cfg_recog, SETTINGS.recog_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#TO BE RELOCATED
#=======================================================================================
# Variables usadas para la identificacion
verifying = False
attempts = 0
identity = "NicoMacian1"
state = None # Puede ser "verificado" o "rechazado"
#=======================================================================================

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, color):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), color, 3)
    #Si queres mostrar el label y confianza arriba de la imagen, descomenta el bloque
    '''
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #Draw rectangle around label
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255),1)
    #Write label text
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    '''

def writeMessage(frame, frameHeight, frameWidth, mensaje, color, factor=6, colorTexto=(0,0,0)):
    cv.rectangle(frame, (0, frameHeight-60), (frameWidth, frameHeight), color, -1)
    cv.putText(frame, mensaje, (int(frameWidth/factor), frameHeight-20), cv.FONT_HERSHEY_SIMPLEX, 1, colorTexto, 2)                

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
        # print("out.shape : ", out.shape)
        for detection in out:            
            scores = detection[5:]
            #Me parece que las proximas 2 lineas no tienen sentido en el caso que reconocemos solo 1 clase
            #Si no me equivoco en la interpretacion de los resultados
            classId = np.argmax(scores)            
            confidence = scores[classId]

            #El proximo if esta al pedo, es algo para debuggear no mas
            # if detection[4]>confThreshold:
            #     print(detection[4], " - ", confidence, " - th : ", confThreshold)
            #     print(detection)

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
        ancho = width*100/frameWidth
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
                    # Vamos con otro intento         
                    # Dibujamos en pantalla
                    drawPred(classIds[i], confidences[i], left, top, right, bottom, (0, 255, 255))
                    writeMessage(frame, frameHeight, frameWidth, 'Verificando:', (0, 255, 255))
                    cv.putText(frame, str(attempts)+' - '+str(dist), (int(frameWidth/2.1), frameHeight-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                else:
                    drawPred(classIds[i], confidences[i], left, top, right, bottom, (0, 255, 0))
                    writeMessage(frame, frameHeight, frameWidth, identity, (0, 255, 0), factor=3)
                    state = 'verificado'
            else:
                # identidad rechazada
                drawPred(classIds[i], confidences[i], left, top, right, bottom, (0, 0, 255))
                writeMessage(frame, frameHeight, frameWidth, 'Identidad no verificada', (0, 0, 255), factor=5)
                state = 'rechazado'
            
        else:
            verifying = False
            writeMessage(frame, frameHeight, frameWidth, 'Acercate mas a la camara', (255, 255, 255))
    elif len(indices)>1:
        verifying = False
        writeMessage(frame, frameHeight, frameWidth, 'Mas de una cara detectada', (0, 0, 255))
    else:
        verifying = False
        writeMessage(frame, frameHeight, frameWidth, 'No se detecto una cara', (0, 0, 255))
                        

#Esto tiene que ver con los argumentos que comente al principio
outputFile = "yolo_out_py.avi"

'''
# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

'''

#Si solo usamos la cam directamente, y el bloque anterior esta comentado
cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 416) 
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 416) 
# vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


#No se que es esto, pero ya estaba comentado
# frame_width =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
# frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float
    

while True:        
    # start = time.time()
    # start1 = start

    # get frame from the video
    hasFrame, frame = cap.read()
    frame = frame[32:448, 112:528]
    #Descomentar el proximo bloque si lo de los argumentos esta descomentados arriba
    '''
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break
    '''

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # end = time.time()
    # print("antes de procesar: {}".format(end-start))
    
    # Presiona espacio para reiniciar
    key = cv.waitKey(1)
    if key == 32:
        state = None
        verifying = False

    if state is None:
        # start = time.time()
        postprocess(frame, outs)
        # end = time.time()
        # print("procesamiento: {}".format(end-start))
    elif state=='verificado':
        writeMessage(frame, frame.shape[0], frame.shape[1], identity, (0, 255, 0), factor=3)
    elif state=='rechazado':
        writeMessage(frame, frame.shape[0], frame.shape[1], 'Identidad no verificada', (0, 0, 255), factor=5)

    # start = time.time()

    #El siguiente bloque muestra tiempo de inferencia.
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Tiempo de inferencia: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    #Yo descomente la linea siguiente, le puse 416 pq asumi que era nuestro input, no se la vdd
    #Comentada o no es lo mismo jeje saludos
    # final_frame = cv.resize(frame, (416, 416))             # Resize image 
    # cv.imshow('Sistema de deteccion facial', final_frame)
    cv.imshow('Sistema de deteccion facial', frame)

    # end = time.time()
    # print("despues de procesar: {}".format(end-start))
    # print('tiempo de vuelta: {}'.format(end-start1))
    # print('--------------------------------')
    # Descomentar si argumentos habilitados
    '''
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))
    '''
    #Comentar si argumentos habilitados
    # vid_writer.write(frame.astype(np.uint8))