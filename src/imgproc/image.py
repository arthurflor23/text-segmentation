import imgproc.cpp as cpp
import os


class Image():
	def __init__(self, src, out, cpp_compile):
		self.name = os.path.basename(src).split(".")[0]
		self.ext = ".png"
		self.src = src
		self.out = os.path.join(out, self.name, "")

		self.compile = cpp_compile
		self.words = []

	def process(self):

		if self.compile:
			cpp.compile()

		cpp.segmentation(self.src, self.out, self.name, self.ext)

		# processamento terminado...
		# words = leitura das imagens das palavras (../out/$FILE/words/*.png)
