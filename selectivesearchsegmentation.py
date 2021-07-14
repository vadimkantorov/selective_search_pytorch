import cv2

class SelectiveSearchSegmentationStrategyMultipleImpl:
	def __init__(self):
		self.strategies = []
		self.weights = []
		self.weights_total = 0.0

	def addStrategy(self, g, weight):
		self.strategies.append(g)
		self.weights.append(weight)
		self.weights_total += weight

	@staticemthod
	def createSelectiveSearchSegmentationStrategyMultiple(s1, s2 = None, s3 = None, s4 = None):
		s = SelectiveSearchSegmentationStrategyMultipleImpl()
		
		if s2 is not None and s3 is not None and s4 is not None:
			s.addStrategy(s1, 0.25)
			s.addStrategy(s2, 0.25)
			s.addStrategy(s3, 0.25)
			s.addStrategy(s4, 0.25)

		else if s2 is not None and s3 is not None:
			s.addStrategy(s1, 0.3333)
			s.addStrategy(s2, 0.3333)
			s.addStrategy(s3, 0.3333)

		else if s2 is not None:
			s.addStrategy(s1, 0.5)
			s.addStrategy(s2, 0.5)

		else:
			s.addStrategy(s1, 1.0)

		
		return s


def SelectiveSearchSegmentationStrategyColorImpl(r1, r2, img, regions, sizes):
	img_planes = cv2.split(img)
	
	min_val, max_val, *_ = cv2.minMaxLoc(regions)
	nb_segs = int(max_val) + 1

	histogram_size = histogram_bins_size * img.channels
	histograms = cv2.Mat(nb_segs, histogram_size)

	for r in range(nb_segs):
		for p in range(img.channels):
			histograms[r][p] = cv2.calcHist(img_planes[p], nimages = 1, channels = [0], mask = regions == r, dims = 1, histSize = [25], ranges = [0, 256])
		
		hisotgrams[r] /= np.sum(histograms[r])

	size_r1, size_r2 = sizes[r1], sizes[r2]
	
	h1, h2 = histograms[r1], histograms[r2]
	return sum(min(h1[i], h2[i]) for i in range(histogram_size))
	# histograms[r1] = histograms[r2] = [ (h1[i] * size_r1 + h2[i] * size_r2) / (size_r1 + size_r2) for i in range(histogram_size) ]

def SelectiveSearchSegmentationStrategySizeImpl(r1, r2, img, regions, sizes):
	size_image = img.rows * img.cols
	size_r1, size_r2 = sizes[r1], sizes[r2]

	return max(min(1.0 - float(size_r1 + size_r2) / float(size_image), 1.0), 0.0)

def SelectiveSearchSegmentationStrategyFillImpl(r1, r2, img, regions, sizes):
	size_image = img.rows * img.cols
	min_val, max_val, *_ = cv2.minMaxLoc(regions)
	nb_segs = int(max_val) + 1
	
	points = [[] for k in range(nb_segs)]
	for i in range(regions.rows):
		p = regions[i]
		for j in range(regions.cols): 
			points[p[j]].append( (j, i) )

	bounding_rects = [cv2.boundingRect(points[seg]) for seg in range(nb_seg)]

	size_r1, size_r2 = sizes[r1], sizes[r2]
	
	bounding_rect_size = (bounding_rects[r1] | bounding_rects[r2]).area()
	
	return max(min(1.0 - float(bounding_rect_size - size_r1 - size_r2) / float(size_image), 1.0), 0.0)
	# bounding_rects[r1] = bounding_rects[r2] = bounding_rects[r1] | bounding_rects[r2]

def SelectiveSearchSegmentationStrategyTextureImpl(r1, r2, img, regions, sizes):
	img_planes = cv2.split(img)
	
	min_val, max_val, *_ = cv2.minMaxLoc(regions)
	nb_segs = int(max_val) + 1

	range_ = [0.0, 256.0]
	histogram_bins_size = 10
	histogram_size = histogram_bins_size * img.channels * 8

	histograms = cv2.Mat(nb_segs, histogram_size)

	img_gaussians = []

	for p in range(img.channels):
		# X, no rot
		tmp_gradiant = cv2.Scharr(img_planes[p], cv2.CV_32F, 1, 0)
		tmp_gradiant_pos = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO)
		tmp_gradiant_neg = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO_INV)
		img_gaussians.extend([tmp_gradiant_pos.clone(), tmp_gradiant_neg.clone()])
		
		# Y, no rot
		tmp_gradiant = cv2.Scharr(img_planes[p], cv2.CV_32F, 0, 1)
		tmp_gradiant_pos = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO)
		tmp_gradiant_neg = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO_INV)
		img_gaussians.extend([tmp_gradiant_pos.clone(), tmp_gradiant_neg.clone()])

		center = (img.cols / 2.0, img.rows / 2.0)
		rot = cv2.getRotationMatrix2D(center, 45.0, 1.0)
		bbox = cv2.RotatedRect(center, img.size(), 45.0).boundingRect()
		rot[0, 2] += bbox.width/2.0 - center.x
		rot[1, 2] += bbox.height/2.0 - center.y
		img_plane_rotated = cv2.warpAffine(img_planes[p], rot, bbox.size())

		# X, rot
		tmp_gradiant = cv2.Scharr(img_plane_rotated, cv2.CV_32F, 1, 0)
		center = (int(img_plane_rotated.cols / 2.0), int(img_plane_rotated.rows / 2.0))
		rot = cv2.getRotationMatrix2D(center, -45.0, 1.0)
		bbox2 = cv2.RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect()
		tmp_rot = cv2.warpAffine(tmp_gradiant, rot, bbox2.size())


		start_x, start_y = max(0, (bbox.width - img.cols) / 2), max(0, (bbox.height - img.rows) / 2)
		
		tmp_gradiant = tmp_rot(Rect(start_x, start_y, img.cols, img.rows))
		tmp_gradiant_pos = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO)
		tmp_gradiant_neg = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO_INV)
		img_gaussians.extend([tmp_gradiant_pos.clone(), tmp_gradiant_neg.clone()])

		// Y, rot
		tmp_gradiant = cv2.Scharr(img_plane_rotated, cv2.CV_32F, 0, 1)
		center = (int(img_plane_rotated.cols / 2.0), int(img_plane_rotated.rows / 2.0))
		rot = cv2.getRotationMatrix2D(center, -45.0, 1.0)
		bbox2 = cv2.RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect()
		tmp_rot = cv2.warpAffine(tmp_gradiant, rot, bbox2.size())

		start_x, start_y = max(0, (bbox.width - img.cols) / 2), max(0, (bbox.height - img.rows) / 2)
		
		tmp_gradiant = tmp_rot(Rect(start_x, start_y, img.cols, img.rows))
		tmp_gradiant_pos = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO)
		tmp_gradiant_neg = cv2.threshold(tmp_gradiant, 0, 0, cv2.THRESH_TOZERO_INV)
		img_gaussians.extend([tmp_gradiant_pos.clone(), tmp_gradiant_neg.clone()])
	
	for i in range(img.channels() * 8):
		hmin, hmax, *_ = cv2.minMaxLoc(img_gaussians[i])
		img_gaussians[i] = img_gaussians[i].convertTo(tmp, cv2.CV_8U, (range_[1] - 1) / (hmax - hmin), -(range_[1] - 1) * hmin / (hmax - hmin));
	
	
	totals = [0] * nb_seg
	tmp_histograms = cv2.Mat_zeros(nb_segs, histogram_size)

	for x in range(regions.total):
		region = regions[x]
		histogram = tmp_histograms[region]

		for p in range(img.channels):
			for i in range(8):
				val = int(img_gaussians[p * 8 + i][x])
				bin = int(float(val) / (range[1] / histogram_bins_size))
				histogram[(p * 8 + i) * histogram_bins_size + bin] += 1
				totals[region] += 1

	for r in range(nb_segs):
		for h_pos2 in range(histogram_size):
			histogram[r][h_pos2] = float(tmp_histogram[r][h_pos2]) / float(totals[r])
	

	h1, h2 = histograms[r1], histograms[r2]
	return sum(min(h1[i], h2[i]) for i in range(histogram_size))
	# histograms[r1] = histograms[r2] = [ (h1[i] * size_r1 + h2[i] * size_r2) / (size_r1 + size_r2) for i in range(histogram_size) ]
