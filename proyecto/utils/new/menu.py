import keyboard
from proyecto.utils.recognition.simple import clear

selected = 0

def show_menu(db_list):
    clear()
    print('Sistema de verificacion facial de identidad')
    print('===========================================\n')
    print("Seleccion su identidad: ")
    for num, name in enumerate(db_list, start=1):
        print("{1} {2}. {0}".format(name, ">" if db_list[selected] == name else " ", num))

def up(db_list):
    global selected
    if selected == 0:
        return
    selected -= 1
    show_menu(db_list)

def down(db_list):
    global selected
    if selected == len(db_list)-1:
        return
    selected += 1
    show_menu(db_list)

def initialize_menu(database):
    db_list = list(database)
    db_list.append('exit')
    show_menu(db_list)
    keyboard.add_hotkey('up', up, args=[db_list])
    keyboard.add_hotkey('down', down, args=[db_list])
    keyboard.wait('enter')
    keyboard.unhook_all()
    print('Se identificara {}...'.format(db_list[selected]))
    #input() # Clear enter key press
    return db_list[selected]