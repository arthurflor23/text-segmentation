import os


class Image():
	def __init__(self, src, out):
		self.name = os.path.basename(src).split(".")[0]
		self.ext = ".png"
		self.src = src
		self.out = os.path.join(out, self.name, "")
		self.words = []
