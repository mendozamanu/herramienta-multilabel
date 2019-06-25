# coding=utf-8
from subprocess import STDOUT, check_call
import os
def main(argv):
    try:
        filename = argv.pop(0)
    except IndexError:
        print("usage: aptreqs.py REQ_FILE")
    else:
	with open(filename, 'r') as f:
            for line in f:
		if not os.path.exists('./tmp/'):
			os.makedirs('./tmp/')
		if os.path.isfile('./tmp/stdout'):
			os.remove('./tmp/stdout')
		try:
			print "Instalando: "+str(line.strip()+"...")
			check_call(['apt-get', 'install', '-y', line.strip()],
		     		stdout=open('./tmp/stdout','ab'), stderr=STDOUT)
		except:
			print "Se ha producido un error en la llamada, compruebe el archivo /tmp/stdout para más información"

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
 
