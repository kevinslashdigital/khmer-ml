import os
import subprocess
import time

class KhmerSegment:
	def __init__(self):
		self.PATH = '/tmp/km-5tag-seg-1.0'

	def add_to_file(self, text):
		seconds = str(int(time.time() * 1000000))
		file_name = seconds + '.txt'
		file = self.PATH + '/sample/' + file_name
		f = open(file, 'w')
		f.write(text)
		f.close()
		return file_name

	def read_file(file):
		f = open(file, 'r')
		content = f.read()
		f.close()
		return content

	def remove_file(file):
		try:
			os.remove(file)
		except OSError:
			pass

	def run_command(command):
		FNULL = open(os.devnull, 'w')
		return subprocess.call(command, stdout = FNULL, stderr=subprocess.STDOUT, shell=True)
