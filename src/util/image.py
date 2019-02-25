from segmentation import binarize, line, word
import environment as env
import util.plot as plot
import numpy as np
import cv2 as cv
import os


class Image():
	def __init__(self, i_path):
		self.name = os.path.basename(i_path).split(".")[0]
		self.file_ext = ".png"

		self.img = cv.imread(i_path, cv.IMREAD_GRAYSCALE)
		self.binary = None
		
		cv.imwrite(self.file_path_out(), self.img)

	def file_path_out(self, ext=None, *args):
		path = os.path.join(env.OUT_PATH, self.name, *args)
		os.makedirs(path, exist_ok=True)

		ext = "" if ext is None else "_" + ext
		return os.path.join(path, self.name + ext + self.file_ext)

	def threshold(self, method):
		if method == "su":
			self.binary = binarize.su(self.img)
		elif method == "suplus":
			self.binary = binarize.su_plus(self.img)
		elif method == "sauvola":
			self.binary = binarize.sauvola(self.img, [127, 127], 127, 0.1)
		else:
			self.binary = binarize.otsu(self.img)

		cv.imwrite(self.file_path_out("1_binary"), self.binary)

	def segment(self):
		l = line.Segmentation(self.binary)

    	# find letters contours
		l.find_contours()
		plot.rects(self.file_path_out("2_contours"), self.binary, l.contours)

		# divide image into vertical chunks
		l.divide_chunks()
		plot.chunks(self.file_path_out("chunk#", "chunks"), l.chunks)
		plot.chunks_histogram(self.file_path_out("3_histogram"), l.chunks)

		# get initial lines
		l.get_initial_lines()
		plot.image_with_lines(self.file_path_out("4_initial_lines"), self.binary, l.initial_lines)

		try:
			# get initial line regions
			l.generate_regions()

			# repair initial lines and generate the final line regions
			l.repair_lines()

			# generate the final line regions
			l.generate_regions()
		except:
			pass

		plot.image_with_lines(self.file_path_out("5_final_lines"), self.binary, l.initial_lines)

		# get lines to segment
		img_lines = l.get_regions()
		plot.lines(self.file_path_out("line#", "lines"), img_lines)
