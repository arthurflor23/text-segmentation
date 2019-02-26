from glob import glob as glob
from imgproc.image import Image
import argparse
import os

SRC_PATH = os.path.join("..", "data")
OUT_PATH = os.path.join("..", "out")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", default=False)
	args = parser.parse_args()

	# images = sorted(glob(os.path.join(SRC_PATH, "*.png")))
	images = sorted(glob(os.path.join(SRC_PATH, "005.png")))

	for src in images:
		im = Image(src, OUT_PATH, args.c)
		im.process()


if __name__ == '__main__':
	main()
