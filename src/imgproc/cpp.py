import os

### segmentation ###
SEG_PATH = os.path.join("imgproc", "segmentation")
SEG_MAIN = os.path.join(SEG_PATH, "main.cpp")
SEG_OUT = os.path.join(".", "preprocess.out")

SEG_BIN = os.path.join(SEG_PATH, "Binarization.cpp")
SEG_LINE = os.path.join(SEG_PATH, "LineSegmentation.cpp")

SEG_COMPILE = ("g++ %s %s %s -o %s `pkg-config --cflags --libs opencv4`" % (SEG_MAIN, SEG_BIN, SEG_LINE, SEG_OUT))

def compile():

	if os.system(SEG_COMPILE) != 0:
		print("Segmentation compile error")
		exit()


def segmentation(src, out, name, ext):

	execute_cmd = ("%s %s %s %s %s" % (SEG_OUT, src, out, name, ext))

	if os.system(execute_cmd) != 0:
		print("Image segmentation error: %s" % src)
		exit()
