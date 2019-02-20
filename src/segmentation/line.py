import environment as env
import cv2 as cv
import numpy as np

class Chunk():
	def __init__(self, i, c, w, chunk):
		self.index = i
		self.start_col = c
		self.width = w
		self.img = chunk.copy()
		self.avg_height = 0
		self.avg_white_height = 0
		self.lines_count = 0
		self.histogram = np.zeros(chunk.shape[0], dtype=np.uint8)
		self.valleys = None
		self.peaks = None

		self.calculate_histogram()

	def calculate_histogram(self):
		blur = cv.medianBlur(self.img, 5)
		black_count, current_height, current_white_count = 0, 0, 0
		white_lines_count, white_spaces = 0, []

		for index, line in enumerate(blur):
			black_count = 0

			for col in line:
				if (col == 0):
					black_count += 1
					self.histogram[index] += 1

			if black_count:
				current_height += 1
				if current_white_count:
					white_spaces.append(current_white_count)
			else:
				current_white_count += 1
				if current_height:
					self.lines_count += 1
					self.avg_height += current_height
				current_height = 0

		# print(self.histogram)


def plot_histograms(chunks):

	con = None

	for c in chunks:
		temp = np.ones(c.img.shape, dtype=np.uint8) * 255

		for row, _ in enumerate(temp):
			for col, _ in enumerate(temp[row]):
				if c.histogram[row] > col:
					temp[row, col] = 0

		con = temp if con is None else np.concatenate([con, temp], axis=1)

	return con


def segment(im):
	### find letters contours
	contours, _ = cv.findContours(im.binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
	drawing = cv.cvtColor(im.binary.copy(), cv.COLOR_GRAY2BGR)
	bound_rect = []

	for contour in contours:
		startX, startY, endX, endY = cv.boundingRect(contour)

		if (startX == 0) and (startY == 0):
			continue

		bound_rect.append(np.array([startX, startY, endX, endY]))
		cv.rectangle(drawing, (startX, startY), (startX+endX, startY+endY), (0,0,255), 2)

	cv.imwrite(im.file_path_out(ext="contours"), drawing)

	### divide image into vertical chunks
	chunks = []
	chunks_number = 20
	chunk_width = im.binary.shape[1] // chunks_number

	for index, c in enumerate(range(0, im.binary.shape[1], chunk_width)):
		chunk = Chunk(index, c, chunk_width, im.binary[0:im.binary.shape[0], c:c+chunk_width])
		chunks.append(chunk)

		# cv.imwrite(im.file_path_out(ext="chunk_"+str(index+1)), chunk.img)

	cv.imwrite(im.file_path_out(ext="histogram"), plot_histograms(chunks))

	### get initial lines
	number_of_heights, valleys_min_abs_dist = 0, 0
	
	# for i in range(5):
	# 	print(chunks[i].histogram)

	# Get the histogram of the first CHUNKS_TO_BE_PROCESSED and get the overall average line height.
    # for (int i = 0; i < CHUNKS_TO_BE_PROCESSED; i++) {
    #     int avg_height = this->chunks[i]->find_peaks_valleys(map_valley);
    #     if (avg_height) number_of_heights++;
    #     valleys_min_abs_dist += avg_height;
    # }

	return []

# // Get initial lines.
# this->get_initial_lines();
# this->save_image_with_lines(OUT_PATH+"Initial_Lines.jpg");

# // Get initial line regions.
# this->generate_regions();

# // Repair initial lines and generate the final line regions.
# this->repair_lines();

# // Generate the final line regions.
# this->generate_regions();

# this->save_image_with_lines(OUT_PATH+"Final_Lines.bmp");

# //is neccesary to use bitmap or tiff for componente labeling
# this->labelImage(OUT_PATH+"labels.bmp");

# line_segmentation.save_lines_to_file(lines);
