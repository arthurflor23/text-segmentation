from glob import glob as glob
import environment as env
import argparse
import imgproc
import os

def main():
	parser = argparse.ArgumentParser()
	### preprocess actions ###
	parser.add_argument("-c", "--compile", help="compile cpp file preprocess", action="store_true", default=False)
	parser.add_argument("-p", "--preprocess", help="execute cpp file preprocess", action="store_true", default=False)

	### data actions ###
	parser.add_argument("-i", "--image", help="use specific image in data path", type=str, default=None)
	parser.add_argument("-d", "--dataset", help="use specific dataset in data path", type=str, default=None)

	### model actions ###
	parser.add_argument("-t", "--train", help="train model", action="store_true", default=False)
	parser.add_argument("-e", "--evaluate", help="evaluate model", action="store_true", default=False)
	args = parser.parse_args()


	### data functions ###
	if args.dataset:
		images = sorted(glob(os.path.join(env.p_SRC, "*.png"))) ## CHANGE
	elif args.image:
		images = sorted(glob(os.path.join(env.p_SRC, args.image)))
	else:
		images = sorted(glob(os.path.join(env.p_SRC, "*.png")))


	### preprocess functions ###
	if args.compile or (args.preprocess and not os.path.exists(env.fn_CPP_OUT)):
		imgproc.compile()

	if args.preprocess:
		imgproc.execute(images)


	### model functions ###
	if args.dataset and args.train:
		print("train model")

	if args.dataset and args.evaluate:
		print("evaluate model")


if __name__ == '__main__':
	main()
