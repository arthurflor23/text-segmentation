from glob import glob as glob
from util.image import Image
import environment as env
import cv2 as cv
import os


def main():

#	images = sorted(glob(os.path.join(env.SRC_PATH, "*.png")))
	images = sorted(glob(os.path.join(env.SRC_PATH, "temp.png")))

	for i_path in images:
		### read data
		im = Image(i_path)

		### preprocessing
		#im.threshold("su")
		#im.threshold("suplus")
		im.threshold("sauvola")
		#im.threshold("otsu")

		im.segment()


if __name__ == '__main__':
	main()
