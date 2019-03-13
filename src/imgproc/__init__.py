from glob import glob as glob
import environment as env
import multiprocessing
import sys
import os


class Image():
	def __init__(self, src):
		self.src = src
		self.src_pp = src.replace(env.p_SRC, env.p_SRC_PP)
		self.out = src.replace(env.p_SRC, env.p_OUT)
		self.logged = "1" ## 0: False, 1: True
		self.name = os.path.basename(src).split(".")[0]


def compile():
	cpp = " ".join(sorted(glob(os.path.join(env.p_CPP_FILES, "*.cpp"))))
	cmd = "g++ %s -o %s `pkg-config --cflags --libs opencv4`" % (cpp, env.fn_CPP_OUT)

	if os.system(cmd) != 0:
		sys.exit("Preprocess compile error")


def execute(images):
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(__execute__, images)
	pool.close()
	pool.join()


def __execute__(image):
	im = Image(image)	
	cmd = " ".join([env.fn_CPP_OUT, im.src, im.src_pp, im.out, im.logged])

	if os.system(cmd) != 0:
		sys.exit("Image process error: %s" % im.src)

