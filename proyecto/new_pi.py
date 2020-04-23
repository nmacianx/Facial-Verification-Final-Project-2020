from os.path import isfile
import cv2 as cv
import sys

import proyecto.data.settings as SETTINGS
from proyecto.utils.new.side_functions import handleCreation
from proyecto.utils.recognition.simple import clear, getIdName
from proyecto.utils.verification.verificator import initialize
from proyecto.utils.recognition.side_functions import getOutputsNames, processNetworkOutput, processFaces, handleNotVerifying
from picamera.array import PiRGBArray
from picamera import PiCamera

def createIdentity():
    # Get verification model initialized and database with stored faces embeddings
    verification_model, database = initialize() 
    id_name = getIdName(database)   # Get user name via input
    # Initialize face detection network model
    net = cv.dnn.readNetFromDarknet(SETTINGS.cfg_recog, SETTINGS.recog_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # Set variables used for managing state
    state = None    
    cv.namedWindow(SETTINGS.WIN_NAME)   # Create a named window
    camera = PiCamera(resolution=(640, 480), framerate=30)
    rawCapture = PiRGBArray(camera, size=(640, 480))
    for x in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Get frame from the video feed and resize to get the center 416x416
        frame = x.array    
        frame = frame[32:448, 112:528]  # Must be changed if camera resolution is not 480x640 
        # Get keyboard input, if enter (13) was pressed then quit 
        key = cv.waitKey(1)
        if key == 13:
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
                if key == 32:
                    state = 'save'
                state = handleCreation(frame, face, position, state, id_name, verification_model, database)
            elif len(indices) > 1:  # More than one face detected
                handleNotVerifying(frame, state, option='many')
            else:       # No face was detected
                handleNotVerifying(frame, state, option='nobody')
        else:
            handleNotVerifying(frame, state, identity=id_name)   
        cv.imshow(SETTINGS.WIN_NAME, frame)
    # Release video feed to end program
    cap.release()
    cv.destroyAllWindows()

createIdentity()