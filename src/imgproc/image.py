from imgproc.preprocess import PreProcess
import os


class Image():
	def __init__(self, src, out):
		self.name = os.path.basename(src).split(".")[0]
		self.ext = ".png"
		self.src = src
		self.out = os.path.join(out, self.name, "")

		self.cpp = PreProcess()
		self.words = []

	def preprocess(self, cpp_compile):

		if cpp_compile or not os.path.exists(self.cpp.out):
			self.cpp.compile()

		self.cpp.execute((self.src, self.out, self.name, self.ext))

		# processamento terminado...
		# words = leitura das imagens das palavras (../out/$FILE/words/*.png)
