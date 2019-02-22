from glob import glob as glob
from util.image import Image
import environment as env
import cv2 as cv
import os


def main():

	# images = sorted(glob(os.path.join(DATA_PATH, "*.png")))
	images = sorted(glob(os.path.join(env.SRC_PATH, "013.png")))

	for image in images:
		### read data
		im = Image(image, cv.imread(image, cv.IMREAD_GRAYSCALE))
		cv.imwrite(im.file_path_out(), im.img)

		### preprocessing
		# im.threshold("su")
		# im.threshold("suplus")
		# im.threshold("sauvola")
		im.threshold("otsu")
		cv.imwrite(im.file_path_out("1_binary"), im.binary)

		im.segment()


if __name__ == '__main__':
	main()
