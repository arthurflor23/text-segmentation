import cv2 as cv
import numpy as np


def rects(path, img, contours):
	drawing = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)

	for (startX, startY, endX, endY) in contours:
		cv.rectangle(drawing, (startX, startY),
                    (startX+endX, startY+endY), (0, 0, 255), 2)

	cv.imwrite(path, drawing)


def chunks(path, chunks):
	for index, chunk in enumerate(chunks):
		cv.imwrite(path.replace("#", str(index+1)), chunk.img)


def chunks_histogram(path, chunks):
	con = None

	for chunk in chunks:
		temp = np.ones(chunk.img.shape, dtype=np.uint8) * 255

		for row, _ in enumerate(temp):
			for col, _ in enumerate(temp[row]):
				if chunk.histogram[row] > col:
					temp[row, col] = 0

		con = temp if con is None else np.concatenate([con, temp], axis=1)

	cv.imwrite(path, con)


def image_with_lines(path, img, initial_lines):
	image = img.copy()

	for line in initial_lines:
		last_row = -1

		for x, y in line.points:
			image[x, y] = 0

			if (last_row != -1) and (x != last_row):
				for h in range(min(last_row, x), max(last_row, x)):
					image[h, y] = 0
			last_row = x

	cv.imwrite(path, image)

def lines(path, lines):
	for index, line in enumerate(lines):
		cv.imwrite(path.replace("#", str(index+1)), line)
