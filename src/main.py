from glob import glob as glob
from functools import partial
from imgproc import Image, PreProcess
import multiprocessing
import argparse
import os

src_path = os.path.join("..", "data")
out_path = os.path.join("..", "out")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", default=False)
	args = parser.parse_args()

	pp = PreProcess()
	pp.compile(args.c)

	# images = sorted(glob(os.path.join(src_path, "*.png")))
	images = sorted(glob(os.path.join(src_path, "009.png")))

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(partial(foo, out=out_path, pp=pp), images)
	pool.close()
	pool.join()


def foo(image, out, pp):
	im = Image(image, out)
	pp.execute(im.src, im.out, im.name, im.ext)


if __name__ == '__main__':
	main()
