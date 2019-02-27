import os


class PreProcess():

	def __init__(self):
		self.path = os.path.join("imgproc", "cpp")
		self.main = os.path.join(self.path, "main.cpp")
		self.out = os.path.join(".", "preprocess.out")

		self.binarize_cpp = os.path.join(self.path, "Binarization.cpp")
		self.line_seg_cpp = os.path.join(self.path, "LineSegmentation.cpp")

		self.compile_cmd = "g++ %s %s %s -o %s `pkg-config --cflags --libs opencv4`"
		self.execute_cmd = "%s %s %s %s"

		### use light distribution ###
		self.light_parameter = "true"

		### niblack = 0 | sauvola = 1 | wolf = 2 | otsu = 3 ###
		self.threshold_parameter = "2"

	def compile(self):
		cmd = (self.compile_cmd % (self.main, self.binarize_cpp, self.line_seg_cpp, self.out))

		if os.system(cmd) != 0:
			print("Preprocess compile error")
			exit()

	def execute(self, default):
		default = str(default)[1:-1].replace(",","")
		cmd = (self.execute_cmd % (self.out, default, self.light_parameter, self.threshold_parameter))

		if os.system(cmd) != 0:
			print("Image preprocess error: %s" % default[0])
			exit()
