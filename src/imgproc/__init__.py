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

		self.binarize_cpp = os.path.join(self.path, "Binarization.cpp")
		self.scanner_cpp = os.path.join(self.path, "Scanner.cpp")
		self.line_seg_cpp = os.path.join(self.path, "LineSegmentation.cpp")
		self.word_seg_cpp = os.path.join(self.path, "WordSegmentation.cpp")

		self.compile_cmd = "g++ %s %s %s %s %s -o %s `pkg-config --cflags --libs opencv4`"

		### use light distribution: False = 0, True = 1 ###
		self.light_parameter = "1"

		### niblack = 0 | sauvola = 1 | wolf = 2 | otsu = 3 ###
		self.threshold_parameter = "1"


	def compile(self, cpp_compile):
		if cpp_compile or not os.path.exists(self.out):
			cmd = (self.compile_cmd % (self.main, self.binarize_cpp, self.scanner_cpp, self.line_seg_cpp, self.word_seg_cpp, self.out))

			if os.system(cmd) != 0:
				print("Preprocess compile error")
				sys.exit()


	def execute(self, src, out, name, ext):
		default = ("%s %s %s %s %s" % (self.out, src, out, name, ext))
		parameters = ("%s %s" % (self.light_parameter, self.threshold_parameter))

		cmd = ("%s %s" % (default, parameters))

		if os.system(cmd) != 0:
			print("Image preprocess error: %s" % src)
			sys.exit()
