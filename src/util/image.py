from segmentation import binarize, line, word
import environment as env
import util.plot as plot
import cv2 as cv
import os


class Image():
	def __init__(self, image_path, img):
		self.name = os.path.basename(image_path).split(".")[0]
		self.file_ext = ".png"
		self.img = img
		self.binary = None

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
			self.binary = binarize.sauvola(self.img, [21, 21], 127, 0.1)
		else:
			self.binary = binarize.otsu(self.img)

	def segment(self):

		l = line.Segmentation(self.binary)

		l.find_contours()
		plot.rects(self.file_path_out("2_contours"), self.binary.copy(), l.contours)

		l.divide_chunks()
		plot.chunks(self.file_path_out("chunk#", "chunks"), l.chunks)
		plot.chunks_histogram(self.file_path_out("3_histogram"), l.chunks)

		l.get_initial_lines()
		plot.image_with_lines(self.file_path_out("4_initial_lines"), self.binary, l.initial_lines)

		l.generate_regions()
		l.repair_lines()
		l.generate_regions()
		plot.image_with_lines(self.file_path_out("5_final_lines"), self.binary, l.initial_lines)

		lines = l.get_regions()
		plot.lines(self.file_path_out("line#", "lines"), self.binary, lines)
