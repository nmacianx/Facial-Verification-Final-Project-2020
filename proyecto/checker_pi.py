import cv2 as cv
import sys
import os.path
import time

import proyecto.data.settings as SETTINGS
from proyecto.utils.verification.verificator import initialize
from proyecto.utils.recognition.side_functions import getOutputsNames, processNetworkOutput, processFaces, handleVerification, handleNotVerifying
from proyecto.utils.new.menu import initialize_menu
from picamera.array import PiRGBArray
from picamera import PiCamera

def run():
    # Get verification model initialized and database with stored faces embeddings
    verification_model, database = initialize()
    identity = None
    while identity != 'exit':
        identity = initialize_menu(database)
        if identity != 'exit':
            id_checker(verification_model, database, identity)
    print('Hasta luego!')

def id_checker(verification_model, database, identity):    
    # Initialize face detection network model
    net = cv.dnn.readNetFromDarknet(SETTINGS.cfg_recog, SETTINGS.recog_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # Set variables used for managing state
    attempts = 0
    state = None # Can be None if verification hasn't started, 'verified' if id
    #            # has been verified, and 'denied' if verification failed,
    #            # 'exit' if program must exit    
    camera = PiCamera(resolution=(640, 480), framerate=30)
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    #while state != 'exit':
    for x in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        start_time = time.time()
        # Get frame from the video feed and resize to get the center 416x416
        frame = x.array
        frame = frame[32:448, 112:528]  # Must be changed if camera resolution is not 480x640 
        # Get keyboard input, if space (32) was pressed then reset, if enter (13) was pressed then quit 
        key = cv.waitKey(1)
        if key == 32:
            state = None
            attempts = 0
        elif key == 13:            
            state = 'exit'
            break   # Break out of for loop
        if state is None:
            # Create a 4D blob from a frame, image needs to be scaled by 1/255, set mean = 0 for 3 channels, swapRB=1
            blob = cv.dnn.blobFromImage(frame, 1/255, (SETTINGS.inp_width, SETTINGS.inp_height), [0,0,0], 1, crop=False)
            # Sets the input to the network and runs the forward pass to get output of the output layers
            net.setInput(blob)    
            outs = net.forward(getOutputsNames(net))
            indices, boxes = processNetworkOutput(frame.shape[0], frame.shape[1], outs)
            if len(indices) == 1:   # If one face was detected
                face, position = processFaces(frame, indices, boxes)
                face = face[0]  # Get the first face in the array (there should be only one as indices=1)
                position = position[0]  # Get the first position element in the array (there should be only one as indices=1)
                state, attempts = handleVerification(frame, face, position, state, attempts, identity, verification_model, database)
            elif len(indices) > 1:  # More than one face detected
                attempts = 0
                handleNotVerifying(frame, state, option='many')
            else:       # No face was detected
                attempts = 0
                handleNotVerifying(frame, state, option='nobody')
        else:
            handleNotVerifying(frame, state, identity=identity)   
        # #Show FPS information
        end_time = time.time()
        if (end_time != start_time):
            label = 'FPS: %.2f' % (1/(end_time - start_time))
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv.imshow('Sistema de deteccion facial', frame)
        rawCapture.truncate(0)
    # Release video feed to end program
    camera.close()
    cv.destroyAllWindows()
    
        


