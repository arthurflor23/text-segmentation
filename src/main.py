from glob import glob as glob
from imgproc.image import Image
from imgproc.preprocess import PreProcess
from functools import partial
import multiprocessing
import argparse
import os

src_path = os.path.join("..", "data")
out_path = os.path.join("..", "out")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", default=False)
	args = parser.parse_args()

	preprocess = PreProcess()
	preprocess.compile(args.c)

	# images = sorted(glob(os.path.join(src_path, "*.png")))
	images = sorted(glob(os.path.join(src_path, "001.png")))

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(partial(foo, out=out_path, pp=preprocess), images)
	pool.close()
	pool.join()


def foo(image, out, pp):
	im = Image(image, out)
	pp.execute(im.src, im.out, im.name, im.ext)


if __name__ == '__main__':
	main()
