from glob import glob as glob
import multiprocessing
import functools
import sys
import os

pn_CPP_FILES = os.path.join("imgproc", "cpp")
fn_CPP_OUT = os.path.join(".", "imgproc.out")

class Image():
	def __init__(self, src, out):
		self.src = src
		self.out = out + src[len(out)+1:len(src)]


def compile():
	cpp = " ".join(sorted(glob(os.path.join(pn_CPP_FILES, "*.cpp"))))
	cmd = "g++ %s -o %s `pkg-config --cflags --libs opencv4`" % (cpp, fn_CPP_OUT)

	if os.system(cmd) != 0:
		sys.exit("Preprocess compile error")


def execute(images, out):
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(functools.partial(__execute__, out=out), images)
	pool.close()
	pool.join()


def __execute__(image, out):
	im = Image(image, out)
	cmd = " ".join([fn_CPP_OUT, im.src, im.out])

	if os.system(cmd) != 0:
		sys.exit("Image process error: %s" % im.src)

