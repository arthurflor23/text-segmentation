from glob import glob as glob
import argparse
import imgproc
import os

pn_SRC = os.path.join("..", "data")
pn_OUT = os.path.join("..", "out")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--compile", help="compile cpp file preprocess", action="store_true", default=False)
	parser.add_argument("-p", "--preprocess", help="execute cpp file preprocess", action="store_true", default=False)
	parser.add_argument("-i", "--image", help="use specific image in data path", type=str, default=None)
	args = parser.parse_args()

	if args.image:
		images = sorted(glob(os.path.join(pn_SRC, args.image)))
	else:
		images = sorted(glob(os.path.join(pn_SRC, "*.png")))

	if args.compile:
		imgproc.compile()

	if args.preprocess:
		imgproc.execute(images, pn_OUT)


if __name__ == '__main__':
	main()
