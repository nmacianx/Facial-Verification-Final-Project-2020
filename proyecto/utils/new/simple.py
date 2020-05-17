from os import system, name

def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def getIdName(database):
    name_ready = False
    name_checked = False
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
    return id_name

# def identify(database):
#     name_checked = False
#     while not name_checked:
#         clear()
#         print('Programa para verificar la identidad de una persona.')
#         print('=====================================================\n')
#         id_name = input('Escriba su nombre: ')
        
#         id_name = id_name.lower().replace(" ", "_")   # Convert input name to lowercase and replace spaces with underscores
#         if id_name not in database and id_name != 'exit':     # Check if name exists in the database
#             print('\nLo sentimos, no existe alguien con ese nombre en el sistema.')
#             input('Presione una tecla para volver a comenzar.')
#         else:
#             name_checked = True
#             print('Se identificara {}...'.format(id_name))
#     return id_name