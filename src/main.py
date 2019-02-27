from glob import glob as glob
from imgproc.image import Image
import argparse
import os

src_path = os.path.join("..", "data")
out_path = os.path.join("..", "out")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", default=False)
	args = parser.parse_args()

	images = sorted(glob(os.path.join(src_path, "*.png")))
	# images = sorted(glob(os.path.join(src_path, "000.png")))

	for src in images:
		im = Image(src, out_path)
		im.preprocess(args.c)


if __name__ == '__main__':
	main()
