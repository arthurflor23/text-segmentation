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

		self.compile_cmd = "\
			g++ %s %s %s %s %s -o %s `pkg-config --cflags --libs opencv4`" % (
                    self.main,
                    self.binarize_cpp,
                    self.scanner_cpp,
                    self.line_seg_cpp,
                    self.word_seg_cpp,
                    self.out
                )

		### use light distribution: False = 0, True = 1 ###
		self.light_parameter = "1"

		### niblack = 0 | sauvola = 1 | wolf = 2 | otsu = 3 ###
		self.threshold_parameter = "1"

	def compile(self, cpp_compile):
		if cpp_compile or not os.path.exists(self.out):
			if os.system(self.compile_cmd) != 0:
				print("Preprocess compile error")
				sys.exit()

	def execute(self, im):
		execute_cmd = "%s %s %s %s %s %s %s" % (
			self.out, im.src, im.out, im.name, im.ext,
			self.light_parameter, self.threshold_parameter
		)

		if os.system(execute_cmd) != 0:
			print("Image preprocess error: %s" % im.src)
			sys.exit()
