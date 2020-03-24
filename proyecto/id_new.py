from os.path import isfile
import cv2 as cv
import sys

import proyecto.data.settings as SETTINGS
from proyecto.utils.new.side_functions import clear, handleCreation
from proyecto.utils.verification.verificator import initialize
from proyecto.utils.recognition.side_functions import getOutputsNames, processNetworkOutput, processFaces, handleNotVerifying

def createIdentity():
    name_ready = False
    name_checked = False
    # Get verification model initialized and database with stored faces embeddings
    verification_model, database = initialize() 
    while not name_checked:
        while not name_ready:
            clear()
            print('Programa para agregar una nueva identidad al sistema.')
            print('=====================================================\n')
            id_name = input('Escriba el nombre de la persona que se cargara: ')
            print('Se cargara {}.'.format(id_name))
            user_input = input('\nEs correcto el nombre? S/N \n')
            if user_input == 's' or user_input == 'S':
                name_ready = True
        id_name = id_name.lower().replace(" ", "_")   # Convert input name to lowercase and replace spaces with underscores
        if id_name in database:     # Check if name exists in the database
            name_ready = False
            print('\nLo sentimos, esa identidad ya esta cargada en el sistema.')
            input('Presione una tecla para volver a comenzar.')
        else:
            name_checked = True
            print('\n\nComienza la captura de la identidad...')
    # Initialize face detection network model
    net = cv.dnn.readNetFromDarknet(SETTINGS.cfg_recog, SETTINGS.recog_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # Set variables used for managing state
    state = None    
    cap = cv.VideoCapture(0)    # Initialize video capture
    cv.namedWindow(SETTINGS.WIN_NAME)   # Create a named window
    while state != 'exit':
        # Get frame from the video feed and resize to get the center 416x416
        hasFrame, frame = cap.read()        
        frame = frame[32:448, 112:528]  # Must be changed if camera resolution is not 480x640 
        # Get keyboard input, if enter (13) was pressed then quit 
        key = cv.waitKey(1)
        if key == 13:
            state = 'exit'
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