class Color:
    BLUE  = '\033[94m'
    GREEN = '\033[92m'
    FAIL  = '\033[91m'
    DONE  = '\033[0m'

def info_msg(msg: str):
	print(Color.GREEN+"[INFO] "+msg+Color.DONE)

def err_msg(msg: str):
	print(Color.RED+"[ERROR] "+msg)
	print("Killing program."+Color.DONE)
	exit()

def done_msg():
	print(Color.BLUE+"[LISTO]"+Color.DONE)