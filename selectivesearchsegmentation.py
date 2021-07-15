import sys
import random
import dataclasses
import copy

import numpy as np
import cv2
import cv2.ximgproc.segmentation

@dataclasses.dataclass(order = True)
class Region:
	rank : float
	id : int
	level : int
	merged_to : int
	bounding_box : tuple

@dataclasses.dataclass(order = True)
class Neighbour:
	similarity : float
	from : int
	to : int
	removed : bool

class RegionSimilarity:
	def __init__(self, features, strategies):
		self.merge = features.merge
		self.strategies = strategies
		self.weights = [1.0 / len(strategies)] * len(strategies)

	def __call__(self, r1, r2):
		return sum(self.weights[i] * s(features, r1, r2) for i, s in enumerate(self.strategies)) / sum(self.weights)


def selectiveSearch(img_bgr, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, fast = True):
	hsv, lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
	
	if fast:
		segmentations = [ cv.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 2, inc_k) ]
		images = [hsv, lab]
	else:
		I = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
		segmentations = [ cv.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 4, inc_k) ]
		images = [hsv, lab, I, hsv[..., 0], np.dstack([R, G, I])]
	
	
	all_regions = []

	for image in images:
		for gs in segmentations:
			img_regions = gs.processImage(image)
			nb_segs = int(img_regions.max()) + 1

			is_neighbour = np.zeros((nb_segs, nb_segs), dtype = np.bool)
			region_areas = np.zeros((nb_segs,), dtype = np.int64)

			previous_p = None
			
			points = [[] for i in range(nb_segs)]

			for i in range(img_regions.shape[0]):
				for j in range(img_regions.shape[1]): 
					p = img_regions[i]
					points[p[j]].append((j, i))
					region_areas[p[j]] += 1

					if i > 0 && j > 0:
						is_neighbour[p[j - 1], p[j]] = is_neighbour[p[j], p[j - 1]] = True
						is_neighbour[previous_p[j], p[j]] = is_neighbour[p[j], previous_p[j]] = True
						is_neighbour[previous_p[j - 1], p[j]] = is_neighbour[p[j], previous_p[j - 1]] = True

				previous_p = p

			bounding_rects = list(map(cv2.boundingRect, points))
			
			features = HandcraftedRegionFeatures(image, img_regions, region_areas)
			
			if fast:
				strategies = [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size])]
			else:
				strategies = [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Size] ) ]
			
			all_regions.extend( region for strategy in strategies for region in hierarchicalGrouping(strategy, is_neighbour, region_areas, nb_segs, bounding_rects) )
			
	
	return list({region.bounding_box : True for region in sorted(all_regions, key = lambda r: r.bounding_box)}.keys())

def bbox_merge(xywh1, xywh2):
	points1, points2 = map(lambda xywh: [(xywh[0], xywh[1]), (xywh[0] + xywh[-2], xywh[1]), (xywh[0], xywh[1] + xywh[-1]), (xywh[0] + xywh[-2], xywh[1] + xywh[-1])], [xywh1, xywh2])
	return cv2.minAreaRect(points1 + points2)

def hierarchicalGrouping(s, is_neighbour, region_areas, nb_segs, bounding_rects):
	region_areas = region_areas.copy()
	regions, similarities = [], []

	for i in range(nb_segs):
		regions.append(Region(id = i, level = i, merged_to = -1, bounding_box = bounding_rects[i]))
		for j in range(i + 1, nb_segs):
			if is_neighbour[i, j]:
				similarities.append(Neighbour(from = i, to = j, similarity = s(i, j), removed = False))

	while len(similarities) > 0:
		similarities.sort()
		p = similarities.pop()

		region_from, region_to = regions[p.from], regions[p.to]
		regions.append(Region(id = min(region_from.id, region_to.id), level = max(region_from.level, region_to.level) + 1, merged_to = -1, bounding_box = bbox_merge(region_from.bounding_box, region_to.bounding_box)))
		regions[p.from].merged_to = regions[p.to].merged_to = len(regions) - 1

		s.merge(region_from.id, region_to.id);

		region_areas[region_from.id] += region_areas[region_to.id]
		region_areas[region_to.id] = region_areas[region_from.id]

		local_neighbours = set()
		for similarity in similarities:
			if similarity.from == p.from or similarity.to == p.from or similarity.from == p.to or similarity.to == p.to:
				from = similarity.to if similarity.from == p.from or similarity.from == p.to else similarity.from
				local_neighbours.add(from)
				similarity.removed = True

		similarities = [sim for sim in similarities if not sim.removed]

		for local_neighbour in local_neighbours:
			similarities.append(Neighbour(from = len(regions) - 1, to = local_neighbour, similarity = s(regions[n.from].id, regions[n.to].id), removed = False))

	for region in regions:
		region.rank = random.random() * region.level

	return regions

class HandcraftedRegionFeatures:
	def __init__(img, regions, region_areas, color_histogram_bins_size = 25, texture_histogram_bins_size = 10):
		img_height, img_width, img_channels = img.shape
		nb_segs = int(regions.max()) + 1
		
		self.region_areas = region_areas
		self.img_area = img_height * img_width
		
		self.color_histograms = np.zeros((nb_segs, img_channels, color_histogram_bins_size))
		for r in range(nb_segs):
			self.color_histograms[r] = cv2.calcHist(img, nimages = 1, channels = list(range(img_channels)), mask = regions == r, dims = 1, histSize = [self.color_histogram_bins_size] * img_channels, ranges = [0, 256])
		self.color_histograms /= np.sum(self.color_histograms, axis = (1, 2))
		
		points = [[] for k in range(nb_segs)]
		for i in range(regions.shape[0]):
			for j in range(regions.shape[1]): 
				points[regions[i, j]].append( (j, i) )
		self.bounding_rects = list(map(cv2.boundingRect, points))

		self.texture_histogram_size = texture_histogram_bins_size * img_channels * 8
		self.texture_histograms = np.zeros((nb_segs, self.texture_histogram_size), dtype = np.float32)

		img_gaussians = []

		for p in range(img_channels):
			img_plane = img[..., p]
			# X, no rot
			tmp_gradient = cv2.Scharr(img_plane, cv2.CV_32F, 1, 0)
			tmp_gradient_pos, tmp_gradient_neg = [cv2.threshold(tmp_gradient, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]]
			img_gaussians.extend([tmp_gradient_pos.clone(), tmp_gradient_neg.clone()])
			
			# Y, no rot
			tmp_gradient = cv2.Scharr(img_plane, cv2.CV_32F, 0, 1)
			tmp_gradient_pos, tmp_gradient_neg = [cv2.threshold(tmp_gradient, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]]
			img_gaussians.extend([tmp_gradient_pos.clone(), tmp_gradient_neg.clone()])

			center = (img_width / 2.0, img_height / 2.0)
			rot = cv2.getRotationMatrix2D(center, 45.0, 1.0)
			bbox = cv2.RotatedRect(center, self.img_area, 45.0).boundingRect()
			rot[0, 2] += bbox.width/2.0 - center[0]
			rot[1, 2] += bbox.height/2.0 - center[1]
			img_plane_rotated = cv2.warpAffine(img_plane, rot, bbox.size())

			# X, rot
			tmp_gradient = cv2.Scharr(img_plane_rotated, cv2.CV_32F, 1, 0)
			center = (int(img_plane_rotated.shape[1] / 2.0), int(img_plane_rotated.shape[0] / 2.0))
			rot = cv2.getRotationMatrix2D(center, -45.0, 1.0)
			bbox2 = cv2.RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect()
			tmp_rot = cv2.warpAffine(tmp_gradient, rot, bbox2.size())

			start_x, start_y = max(0, (bbox.width - img_width) / 2), max(0, (bbox.height - img_height) / 2)
			tmp_gradient = tmp_rot(Rect(start_x, start_y, img_width, img_height))
			tmp_gradient_pos, tmp_gradient_neg = [cv2.threshold(tmp_gradient, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]]
			img_gaussians.extend([tmp_gradient_pos.clone(), tmp_gradient_neg.clone()])

			// Y, rot
			tmp_gradient = cv2.Scharr(img_plane_rotated, cv2.CV_32F, 0, 1)
			center = (int(img_plane_rotated.shape[1] / 2.0), int(img_plane_rotated.shape[0] / 2.0))
			rot = cv2.getRotationMatrix2D(center, -45.0, 1.0)
			bbox2 = cv2.RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect()
			tmp_rot = cv2.warpAffine(tmp_gradient, rot, bbox2.size())

			start_x, start_y = max(0, (bbox.width - img_width) / 2), max(0, (bbox.height - img_height) / 2)
			tmp_gradient = tmp_rot(Rect(start_x, start_y, img_width, img_height))
			tmp_gradient_pos, tmp_gradient_neg = [cv2.threshold(tmp_gradient, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]]
			img_gaussians.extend([tmp_gradient_pos.clone(), tmp_gradient_neg.clone()])
		
		range_1 = 256.0
		for i in range(img_channels * 8):
			hmin, hmax = img_gaussians[i].min(), img_gaussians[i].max()
			img_gaussians[i] = img_gaussians[i].convertTo(tmp, cv2.CV_8U, (range_1 - 1) / (hmax - hmin), -(range_1 - 1) * hmin / (hmax - hmin))
		
		totals = np.zeros((nb_segs,), dtype = np.int32) 
		for x in range(regions.total):
			for p in range(img_channels):
				for i in range(8):
					val = int(img_gaussians[p * 8 + i][x])
					bin = int(float(val) / (range_1 / self.texture_histogram_bins_size))
					self.texture_histograms[regions[x], (p * 8 + i) * self.texture_histogram_bins_size + bin] += 1
					totals[regions[x]] += 1
		self.texture_histograms /= totals[:, None]


	def merge(self, r1, r2):
		self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_areas[r1] + self.color_histograms[r2] * self.region_areas[r2]) / (self.region_areas[r1] + self.region_areas[r2]) 
		self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_areas[r1] + self.texture_histograms[r2] * self.region_areas[r2]) / (self.region_areas[r1] + self.region_areas[r2]) 
		self.bounding_rects[r1] = self.bounding_rects[r2] = bbox_merge(self.bounding_rects[r1], self.bounding_rects[r2])


	def Size(self, r1, r2):
		return max(min(1.0 - float(self.region_areas[r1] + self.region_areas[r2]) / float(self.img_area), 1.0), 0.0)
	
	def Fill(self, r1, r2):
		return max(min(1.0 - float(  bbox_merge(self.bounding_rects[r1], self.bounding_rects[r2]).area() - self.region_areas[r1] - self.region_areas[r2]) / float(self.img_area), 1.0), 0.0)
	
	def Color(self, r1, r2):
		return np.sum(np.minimum(self.color_histograms[r1], self.color_histograms[r2]))

	def Texture(self, r1, r2):
		return np.sum(np.minimum(self.texture_histograms[r1], self.texture_histograms[r2]))

if __name__ == '__main__':
	base_image = cv2.imread(sys.argv[1])
	regions = selectiveSearchFast(base_image)
	print(regions)
