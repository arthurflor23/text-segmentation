from glob import glob as glob
from segmentation import binarize, line
import environment as env
import cv2 as cv
import os

class Image():
	def __init__(self, image_path):
		self.name = os.path.basename(image_path).split(".")[0]
		self.file_ext = ".png"

		self.img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
		self.binary = None
		
	def file_path_out(self, ext=None, *sub):
		path = os.path.join(env.OUT_PATH, self.name, *sub)
		os.makedirs(path, exist_ok=True)

		ext = "" if ext is None else "_" + ext
		return os.path.join(path, self.name + ext + self.file_ext)

def main():

	# images = sorted(glob(os.path.join(DATA_PATH, "*.png")))
	images = sorted(glob(os.path.join(env.SRC_PATH, "013.png")))

	for image in images:
		### read data
		im = Image(image)
		cv.imwrite(im.file_path_out(), im.img)

		### preprocessing (binary)
		# binary = binarize.otsu(gray)
		# binary = binarize.sauvola(gray, [21, 21], 127, 0.1)
		# binary = binarize.su(gray)
		im.binary = binarize.su_plus(im)
		cv.imwrite(im.file_path_out(ext="binary"), im.binary)

		lines = line.segment(im)

		# break


if __name__ == '__main__':
	main()
