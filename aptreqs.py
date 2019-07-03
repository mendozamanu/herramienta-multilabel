# coding=utf-8
from subprocess import STDOUT, check_call
import os
def main(argv):
	try:
		filename = argv.pop(0)
	except IndexError:
		print("usage: aptreqs.py REQ_FILE")
	else:
		if not os.path.exists('./temp/'):
			os.makedirs('./temp/')
		if os.path.isfile('./temp/stdout'):
			os.remove('./temp/stdout')
		
	with open(filename, 'r') as f:
		print "Ejecutando apt-get update..."
		check_call(['apt-get', 'update'],
						   stdout=open('./temp/stdout','ab'), stderr=STDOUT)
		for line in f:
			try:
				print "Instalando: "+str(line.strip()+"...")
				check_call(['apt-get', 'install', '-y', line.strip()],
							stdout=open('./temp/stdout','ab'), stderr=STDOUT)
			except:
				print "Se ha producido un error en la llamada, compruebe el archivo /tmp/stdout para más información"

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
 
