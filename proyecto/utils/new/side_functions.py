from os import system, name
import cv2 as cv
import numpy as np
from proyecto.utils.recognition.side_functions import handleNotVerifying, drawPred, writeMessage
import proyecto.data.settings as SETTINGS
from proyecto.utils.verification.load_data import update_database

def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def handleCreationOutput(frame, state):
    if state is None:
        writeMessage(frame, 'Pulse espacio para capturar la imagen', SETTINGS.GREEN, text_scale = 0.6)
    elif state == 'save':
        writeMessage(frame, 'Presione S/N para confirmar o abortar', SETTINGS.GREEN, text_scale = 0.6)
    elif state == 'confirmed':
        writeMessage(frame, 'Guardando...', SETTINGS.GREEN)  


def handleCreation(frame, face, position, state, identity, model, database):
    face_height = (position[3]-position[1]) * 100 / frame.shape[0]  # Calculate the relative face height (without added padding)
    if face_height < SETTINGS.face_height_max_threshold:
        if face_height > SETTINGS.face_height_min_threshold:
            # Crop frame to get the detected face, with added padding
            part_image = frame[max(0, position[1] - SETTINGS.face_padding):min(frame.shape[0], position[3] + SETTINGS.face_padding), max(0, position[0] - SETTINGS.face_padding):min(frame.shape[1], position[2] + SETTINGS.face_padding)].copy()
            drawPred(frame, position, SETTINGS.GREEN)
            handleCreationOutput(frame, state)            
            if state == 'save': # Should freeze the image to confirm
                cv.imshow(SETTINGS.WIN_NAME, frame)     # Show the image before the capture freezes
                while state != 'confirmed' and state != 'cancelled':
                    key = cv.waitKey(1)
                    if key == 83 or key == 115: #If s or S was pressed
                        state = 'confirmed'
                    elif key == 78 or key == 110: #If n or N was pressed
                        state = 'cancelled'
                if state == 'confirmed':           
                    # NO funcionan las 2 siguientes
                    handleCreationOutput(frame, state)            
                    cv.imshow(SETTINGS.WIN_NAME, frame)
                    #arreglar anteriores
                    update_database(database, model, part_image, identity, SETTINGS.data_dir)  #Update database and save captured image
                    state = 'created'
                    handleNotVerifying(frame, state, identity=identity)   
                else:   # Image capture was cancelled
                    state = None
        else:
            drawPred(frame, position, SETTINGS.RED)
            handleNotVerifying(frame, state, option='small')     
    else:
        drawPred(frame, position, SETTINGS.RED)
        handleNotVerifying(frame, state, option='big')        
    return state
