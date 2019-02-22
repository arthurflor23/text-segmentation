import cv2 as cv
import numpy as np

ID = 0
CHUNKS_NUMBER = 20
CHUNKS_TO_PROCESS = 5

class Segmentation():

	def __init__(self, binary):
		self.binary = binary.copy()
		self.primes = [] 
		self.sieve(int(1e4))

		self.contours = []
		self.chunks = []
		self.initial_lines = []
		self.line_regions = []
		self.map_valley = {}

		self.predicted_line_height = 0
		self.avg_line_height = 0
		self.chunk_width = 0

		self.red_start = 20


	def sieve(self, n):
		flags = np.ones(n, dtype=bool)
		flags[0] = flags[1] = False

		for i in range(2, n):
			if flags[i]:
				flags[i*i::i] = False

		self.primes = np.flatnonzero(flags)


	def add_primes_list(self, n, prob_primes):
		for i in range(len(self.primes)):

			while int(n % self.primes[i]):
				n /= self.primes[i]
				prob_primes[i] += 1


	def find_contours(self):
		contours, _ = cv.findContours(self.binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

		for contour in contours:
			epsilon = 0.03 * cv.arcLength(contour, True)
			approx = cv.approxPolyDP(contour, epsilon, True)
			startX, startY, endX, endY = cv.boundingRect(approx)

			if (startX == 0) and (startY == 0):
				continue

			self.contours.append(np.array([startX, startY, endX, endY]))


	def divide_chunks(self):
		chunk_width = self.binary.shape[1] // CHUNKS_NUMBER

		for index, c in enumerate(range(0, self.binary.shape[1], chunk_width)):
			chunk = Chunk(index, c, chunk_width, self.binary[0:self.binary.shape[0], c:c+chunk_width])
			self.chunks.append(chunk)
	
	def get_initial_lines(self):
		number_of_heights, valleys_min_abs_dist = 0, 0

		for i in range(CHUNKS_TO_PROCESS):
			self.chunks[i].avg_height = self.chunks[i].find_peaks_valleys(self.map_valley)
			
			if (self.chunks[i].avg_height): 
				number_of_heights += 1

			valleys_min_abs_dist += self.chunks[i].avg_height

		valleys_min_abs_dist //= number_of_heights
		self.predicted_line_height = valleys_min_abs_dist
		print("Estimated avg line height:", valleys_min_abs_dist)

		for i in range(CHUNKS_TO_PROCESS-1, -1, -1):
			if not self.chunks[i].valleys: continue

			for valley in self.chunks[i].valleys:
				if valley.used: continue
				valley.used = True

				new_line = Line(valley.valley_id)
				new_line = self.connect_valleys(i-1, valley, new_line, valleys_min_abs_dist)
				new_line.generate_initial_points(self.chunks[0].width, self.binary.shape[1], self.map_valley)

				if (len(new_line.valleys_ids) > 1):
					self.initial_lines.append(new_line)


	def connect_valleys(self, i, current_valley, line, valleys_min_abs_dist):
		if (i <= 0) or (not self.chunks[i].valleys): return line

		connected_to = -1
		min_distance = 100000

		for j, valley in enumerate(self.chunks[i].valleys):
			if valley.used: continue

			dist = current_valley.position - valley.position
			dist = -dist if dist < 0 else dist

			if (min_distance > dist) and (dist <= valleys_min_abs_dist):
				min_distance = dist 
				connected_to = j

		if connected_to == -1: return line

		line.valleys_ids.append(self.chunks[i].valleys[connected_to].valley_id)
		v = self.chunks[i].valleys[connected_to]
		v.used = True

		return self.connect_valleys(i-1, v, line, valleys_min_abs_dist)


	def generate_regions(self):

		self.line_regions = []
		r = Region(None, self.initial_lines[0])
		r.update_region(self.binary, 0)

		self.initial_lines[0].above = r
		self.line_regions.append(r)

		if r.height < (self.predicted_line_height * 2.5):
			self.avg_line_height += r.height
		
		for i in range(len(self.initial_lines)):
			top_line = self.initial_lines[i]
			bottom_line = None if i == len(self.initial_lines)-1 else self.initial_lines[i+1]

			r = Region(top_line, bottom_line)
			res = r.update_region(self.binary, i)

			if top_line:
				top_line.below = r
			if bottom_line:
				bottom_line.above = r
			if not res:
				self.line_regions.append(r)

				if r.height < (self.predicted_line_height * 2.5):
					self.avg_line_height += r.height
		
		if len(self.line_regions) > 0:
			self.avg_line_height //= len(self.line_regions)
			print("Avg line height is", self.avg_line_height)


	def repair_lines(self):

		for line in self.initial_lines:
			column_processed = {}

			for i in range(len(line.points)):
				x, y = line.points[i][0], line.points[i][1]

				if self.binary[x,y] == 255:

					if i == 0: continue
					black_found = False

					if line.points[i-1][0] != line.points[i][0]:

						min_row = min(line.points[i-1][0], line.points[i][0])
						max_row = max(line.points[i-1][0], line.points[i][0])

						for j in range(min_row, max_row+1 and not black_found):
							if self.binary[j, line.points[i-1][1] == 0]:
								x, y = j, line.points[i-1][1]
								black_found = True
					if not black_found: continue

				try:
					if column_processed[y]: continue
				except:
					column_processed[y] = True

				for contour in self.contours:
					tl_x, tl_y, w, h = contour
					br_x, br_y = (tl_x+w), (tl_y+h)

					if (y >= tl_x) and (y <= br_x) and (x >= tl_y) and (x <= br_y):
						if (br_y - tl_y > self.avg_line_height * 0.9): continue
						
						is_component_above = self.component_to_above(line, contour)
						new_row = None

						if not is_component_above:
							new_row = tl_y
							line.min_row_position = min(new_row, line.min_row_position)
						else:
							new_row = br_y
							line.max_row_position = max(new_row, line.max_row_position)
						
						for k in range(tl_x, br_x):
							line.points[k][0] = new_row
						
						i = tl_x + w
						break


	def component_to_above(self, line, contour):
		prob_above_primes = np.zeros(len(self.primes), dtype=np.uint8)
		prob_below_primes = np.zeros(len(self.primes), dtype=np.uint8)
		n = 0

		tl_x, tl_y, w, h = contour
		br_x, br_y = (tl_x+w), (tl_y+h)

		for y in range(tl_x, br_x):
			for x in range(tl_y, br_y):
				if self.binary[x,y] == 255: continue
				n += 1

				contour_point = np.zeros((1,2))
				contour_point[0,0] = x
				contour_point[0,1] = y

				new_prob_above = line.above.bi_variate_gaussian_density(
					contour_point.copy()) if line.above else 0
				new_prob_below = line.below.bi_variate_gaussian_density(
					contour_point.copy()) if line.below else 0

				self.add_primes_list(new_prob_above, prob_above_primes)
				self.add_primes_list(new_prob_below, prob_below_primes)

		prob_above, prob_below = 0, 0

		for k in range(len(prob_above_primes)):
			mini = min(prob_above_primes[k], prob_below_primes[k])

			prob_above_primes[k] -= mini
			prob_below_primes[k] -= mini

			prob_above += prob_above_primes[k] * self.primes[k]
			prob_below += prob_below_primes[k] * self.primes[k]

		return prob_above < prob_below

	def get_regions(self):
		return [r.region for r in self.line_regions] if len(self.line_regions) > 0 else [self.binary]

class Region():

	def __init__(self, top, bottom):
		self.top = top
		self.bottom = bottom
		self.height = 0
		self.row_offset = None
		self.region_id = 0
		self.region = None
		self.mean = None
		self.covariance = None

	def bi_variate_gaussian_density(self, point):
		point[0,0] -= self.mean[0]
		point[0,1] -= self.mean[1]

		point_transpose = np.round(np.transpose(point), 6)
		covariance_inv = np.linalg.inv(self.covariance)

		ret = np.sum(point * covariance_inv * point_transpose)		
		ret *= np.sqrt(np.linalg.det(self.covariance * 2 * np.pi))

		return int(ret)

	def update_region(self, binary_image, region_id):
		self.region_id = region_id

		min_region_row = self.row_offset = 0 if (self.top == None) else self.top.min_row_position
		max_region_row = binary_image.shape[0] if (self.bottom == None) else self.bottom.max_row_position
		
		start = min(min_region_row, max_region_row)
		end = max(min_region_row, max_region_row)
		self.region = np.ones((end-start, binary_image.shape[1]), dtype=np.uint8) * 255

		for c in range(binary_image.shape[1]):
			start = 0 if self.top == None else self.top.points[c][0]
			end = binary_image.shape[0]-1 if self.bottom == None else self.bottom.points[c][0]
		
			if end > start:
				self.height = max(self.height, end-start)

			for i in range(start, end):
				self.region[i-min_region_row, c] = binary_image[i,c]
		
		self.calculate_mean()
		self.calculate_covariance()

		return cv.countNonZero(self.region) == (self.region.shape[0] * self.region.shape[1])

	def calculate_mean(self):
		self.mean, n = np.array([0, 0], dtype=float), 0

		for y in range(self.region.shape[0]):
			for x in range(self.region.shape[1]):
				if self.region[y,x] == 255: continue

				if n == 0:
					self.mean = [y + self.row_offset, x]
				else:
					a = np.round((n-1)/n, 7)
					b = np.round(np.multiply(a, self.mean), 4)
					c = np.round(np.multiply((1.0/n), [y + self.row_offset, x]), 4)
					self.mean = np.round(np.add(b, c), 6)
				n += 1
			
			self.mean = np.round(self.mean, 3)

	def calculate_covariance(self):
		covariance = np.zeros((2,2), dtype=np.float)
		sum_y_squared, sum_x_squared, sum_y_x, n = 0., 0., 0., 0

		for y in range(self.region.shape[0]):
			for x in range(self.region.shape[1]):
				if self.region[y,x] == 255: continue
				
				new_y = y + self.row_offset - self.mean[0]
				new_x = x - self.mean[1]

				sum_y_squared += new_y * new_y
				sum_y_x += new_y * new_x
				sum_x_squared += new_x * new_x
				n += 1
		
		if n:
			covariance[0,0] = sum_y_squared / n
			covariance[0,1] = sum_y_x / n
			covariance[1,0] = sum_y_x / n
			covariance[1,1] = sum_x_squared / n

		self.covariance = covariance.copy()


class Line():

	def __init__(self, initial_valley_id):
		self.above = None
		self.below = None
		self.valleys_ids = [initial_valley_id]
		self.min_row_position = 0
		self.max_row_position = 0
		self.points = []

	def generate_initial_points(self, chunk_width, img_width, map_valley):
		c, previous_row = 0, 0
		self.valleys_ids = sorted(self.valleys_ids)

		if map_valley[self.valleys_ids[0]].chunk_index > 0:
			previous_row = map_valley[self.valleys_ids[0]].position
			self.max_row_position = self.min_row_position = previous_row

			for j in range(map_valley[self.valleys_ids[0]].chunk_index * chunk_width):
				if c == j: 
					c += 1
					self.points.append([previous_row, j])

		for i in self.valleys_ids:
			chunk_index = map_valley[i].chunk_index
			chunk_row = map_valley[i].position
			chunk_start_column = chunk_index * chunk_width

			for j in range(chunk_start_column, (chunk_start_column + chunk_width)):
				self.min_row_position = min(self.min_row_position, chunk_row)
				self.max_row_position = max(self.max_row_position, chunk_row)

				if c == j:
					c += 1
					self.points.append([chunk_row, j])

			if (previous_row != chunk_row):
				previous_row = chunk_row
				self.min_row_position = min(self.min_row_position, chunk_row)
				self.max_row_position = max(self.max_row_position, chunk_row)

		if CHUNKS_NUMBER-1 > map_valley[self.valleys_ids[-1]].chunk_index:
			chunk_index = map_valley[self.valleys_ids[-1]].chunk_index
			chunk_row = map_valley[self.valleys_ids[-1]].position

			for j in range(chunk_index * chunk_width + chunk_width, img_width):
				if c == j:
					c += 1
					self.points.append([chunk_row, j])


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
		self.valleys = []
		self.peaks = []

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

		white_spaces = sorted(white_spaces)

		for x in white_spaces:
			if x > (self.avg_height * 4):
				break
			self.avg_white_height += x
			white_lines_count += 1

		if white_lines_count:
			self.avg_white_height /= white_lines_count

		if self.lines_count:
			self.avg_height /= self.lines_count
		
		self.avg_height = max(30, int(self.avg_height + (self.avg_height / 2)))


	def find_peaks_valleys(self, map_valley):

		for i in range(1, len(self.histogram)-1):

			left_val = self.histogram[i-1]
			centre_val = self.histogram[i]
			right_val = self.histogram[i+1]

			if (centre_val >= left_val) and (centre_val >= right_val):
				if (self.peaks) and (i - self.peaks[-1][0] <= self.avg_height/2) and (centre_val >= self.peaks[-1][1]):
					self.peaks[-1][0] = i
					self.peaks[-1][1] = centre_val
				elif (self.peaks) and (i - self.peaks[-1][0] <= self.avg_height/2) and (centre_val < self.peaks[-1][1]):
					continue
				else:
					self.peaks.append([i, centre_val])

		new_peaks = []
		peaks_average_values = 0

		for peak in self.peaks:
			peaks_average_values += peak[1]
		
		peaks_average_values /= len(self.peaks)

		for peak in self.peaks:
			if peak[1] >= int(peaks_average_values/4):
				new_peaks.append(peak)

		self.peaks = sorted(new_peaks)
		self.lines_count = len(self.peaks)

		for i in range(1, len(self.peaks)):
			min_position = (self.peaks[i-1][0] + self.peaks[i][0]) // 2
			min_value = self.histogram[min_position]

			j = int(self.peaks[i-1][0] + (self.avg_height/2))
			e = int(self.img.shape[0] if i == len(self.peaks) else self.peaks[i][0] - (self.avg_height - 30))

			while j < e:
				valley_black_count = 0

				for l in range(self.width):
					if self.img[j,l] == 0:
						valley_black_count += 1

				if (i == len(self.peaks)) and (valley_black_count <= min_value):
					min_value = valley_black_count
					min_position = j

					if not min_value:
						min_position = min((self.img.shape[0] - 10), (min_position + self.avg_height))
						j = self.img.shape[0]

					elif (min_value != 0) and (valley_black_count <= min_value):
						min_value = valley_black_count
						min_position = j
				j += 1
			
			global ID
			new_valley = Valley(ID, self.index, min_position)
			ID += 1

			self.valleys.append(new_valley)
			map_valley[new_valley.valley_id] = new_valley

		return int(np.ceil(self.avg_height))


class Valley():

	def __init__(self, ID, c_id, p):
		self.valley_id = ID
		self.chunk_index = c_id
		self.position = p
		self.used = False
