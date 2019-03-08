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
	parser.add_argument("--image", type=str, default=None)
	args = parser.parse_args()

	pp = PreProcess()
	pp.compile(args.c)

	if (args.image):
		images = sorted(glob(os.path.join(src_path, args.image)))
	else:
		images = sorted(glob(os.path.join(src_path, "*.png")))

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(partial(foo, out=out_path, pp=pp), images)
	pool.close()
	pool.join()


def foo(im, out, pp):
	pp.execute(Image(im, out))


if __name__ == '__main__':
	main()
