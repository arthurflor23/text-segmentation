from glob import glob as glob
import os
import sys


class Image():
	def __init__(self, src, out):
		self.name = os.path.basename(src).split(".")[0]
		self.ext = ".png"
		self.src = src
		self.out = os.path.join(out, self.name, "")
		self.words = []


class PreProcess():

	def __init__(self):
		self.path = os.path.join("imgproc", "cpp")
		self.main = os.path.join(self.path, "main.cpp")
		self.out = os.path.join(".", "preprocess.out")

		self.cpp = " ".join(sorted(glob(os.path.join(self.path, "*.cpp"))))
		self.compile_cmd = "g++ %s -o %s `pkg-config --cflags --libs opencv4`" % (self.cpp, self.out)

		self.parameters = [
			"1",  # use light distribution: False = 0, True = 1 ###
			"1",  # niblack = 0 | sauvola = 1 | wolf = 2 | otsu = 3 ###
		]


	def compile(self, cpp_compile):
		if cpp_compile or not os.path.exists(self.out):
			if os.system(self.compile_cmd) != 0:
				print("Preprocess compile error")
				sys.exit()

	def execute(self, im):
		execute_cmd = "%s %s %s %s %s %s" % (
			self.out, im.src, im.out, im.name, im.ext,
			" ".join(self.parameters)
		)

		if os.system(execute_cmd) != 0:
			print("Image preprocess error: %s" % im.src)
			sys.exit()
