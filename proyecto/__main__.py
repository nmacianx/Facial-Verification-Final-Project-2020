import os

if __name__ == '__main__':
    if os.name == 'nt':
        from proyecto import checker
        checker.run()
    else:
        from proyecto import checker_pi
        checker_pi.run()