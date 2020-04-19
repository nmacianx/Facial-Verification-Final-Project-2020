import cv2 as cv
import numpy as np
import proyecto.data.settings as SETTINGS
from proyecto.utils.verification.verificator import verify

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(frame, position, color=(0,255,0)):
    # Draw a bounding box
    cv.rectangle(frame, (position[0], position[1]), (position[2], position[3]), color, 3)

def writeMessage(frame, message, background_color, text_color=(0,0,0), text_scale=1):
    text_size = cv.getTextSize(message, cv.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0]   # Get the text size
    cv.rectangle(frame, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), background_color, -1) # Draw the background rectangle
    cv.putText(frame, message, (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 2) # Draw centered text      

def processNetworkOutput(frame_height, frame_width, outs):
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:            
            scores = detection[5:]      # To understand the meaning of each of the outputs of detection, see documentation
            ##########
            ##revisar toda esta parte
            #############
            #Me parece que las proximas 2 lineas no tienen sentido en el caso que reconocemos solo 1 clase
            #Si no me equivoco en la interpretacion de los resultados
            classId = np.argmax(scores)            
            confidence = scores[classId]         
            if confidence > SETTINGS.recog_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, SETTINGS.recog_threshold, SETTINGS.nms_threshold)     
    return indices, boxes

def processFaces(frame, indices, boxes):
    faces = []
    positions = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        part_image = frame[max(0, top - SETTINGS.face_padding):min(frame.shape[0], bottom + SETTINGS.face_padding), max(0, left - SETTINGS.face_padding):min(frame.shape[1], right + SETTINGS.face_padding)]
        faces.append(part_image)
        positions.append([left, top, right, bottom])
    return faces, positions

def handleNotVerifying(frame, state, identity=None, option=None):
    if state == 'verified':
        writeMessage(frame, identity, SETTINGS.GREEN)
    elif state == 'denied':
        writeMessage(frame, 'Identidad no verificada', SETTINGS.RED)
    elif state == 'created':
        writeMessage(frame, 'Bienvenide ' + identity, SETTINGS.GREEN, text_scale=0.7)
    else:
        if option == 'big':
            writeMessage(frame, 'Muy cerca de la camara', SETTINGS.WHITE)
        elif option == 'small':
            writeMessage(frame, 'Muy lejos de la camara', SETTINGS.WHITE)
        elif option == 'nobody':
            writeMessage(frame, 'No se ha detectado una cara', SETTINGS.WHITE, text_scale=0.7)
        elif option == 'many':
            writeMessage(frame, 'Mas de una cara detectada', SETTINGS.WHITE, text_scale=0.7)

def handleVerificationOutput(frame, attempts, identity, dist):
    mensaje = 'Verificando: {} - {:.2f}'.format(attempts, dist)
    writeMessage(frame, mensaje, SETTINGS.YELLOW)

def handleVerification(frame, face, position, state, attempts, identity, model, database):
    face_height = (position[3]-position[1]) * 100 / frame.shape[0]  # Calculate the relative face height (without added padding)
    if face_height < SETTINGS.face_height_max_threshold:
        if face_height > SETTINGS.face_height_min_threshold:
            # Crop frame to get the detected face, with added padding
            part_image = frame[max(0, position[1] - SETTINGS.face_padding):min(frame.shape[0], position[3] + SETTINGS.face_padding), max(0, position[0] - SETTINGS.face_padding):min(frame.shape[1], position[2] + SETTINGS.face_padding)]
            attempts += 1        
            dist = verify(part_image, identity, model, database)     # The function that actually verifies the identity
            if not (dist < SETTINGS.verif_threshold):  # Didn't verify the identity
                if attempts == SETTINGS.max_attempts:
                    state = 'denied'
                    handleNotVerifying(frame, state)
                else:
                    drawPred(frame, position, SETTINGS.YELLOW)                
                    handleVerificationOutput(frame, attempts, identity, dist)
            else:
                drawPred(frame, position, SETTINGS.GREEN)
                state = 'verified'
                handleNotVerifying(frame, state, identity=identity)
        else:
            attempts = 0
            drawPred(frame, position, SETTINGS.RED)
            handleNotVerifying(frame, state, option='small')     
    else:
        attempts = 0
        drawPred(frame, position, SETTINGS.RED)
        handleNotVerifying(frame, state, option='big')        
    return state, attempts