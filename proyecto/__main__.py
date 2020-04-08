import os
from proyecto import checker
from proyecto import checker_pi

if __name__ == '__main__':
    if os.name == 'nt':
        checker.run()
    else:
        checker_pi.run()