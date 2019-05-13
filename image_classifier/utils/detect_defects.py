"""
	@author: Ingrid Navarro
	@date: Feb 5th, 2019
	@brief: Codigo para post-procesar las imagenes obtenidas 
"""
from tqdm import tqdm
from statistics import mode
import numpy as np
import time as t
import itertools
import argparse
import os
import cv2
import re
import math
import datetime as dt
from skimage import feature, exposure

VIEWPOINTS  = ["A", "B"]
KERNEL_SIZE = 3
LINE_ANGLE  = 5 
VOTES       = 100
RHO         = 1
THETA       = np.pi / 180

PAINT_LOWER_LIMIT = 100
PAINT_UPPER_LIMIT = 60 
PAINT_LIMITU = 70
PAINT_LIMITL = 90

def assert_line_ylimit(y):
	return y > PAINT_UPPER_LIMIT and y < PAINT_LOWER_LIMIT

def upper_thresh_paint(y):
	return y < PAINT_LIMITU

def lower_thresh_paint(y):
	return y > PAINT_LIMITL

def get_lines(canny, bgr, sigma=0.33):
	line_limit1 = LINE_ANGLE * np.pi / 180
	lines = cv2.HoughLines(canny, RHO, THETA, VOTES)
	if lines.all() != None:
		for i in range(len(lines)):
			for rho, theta in lines[i]:
				if (rho > 0 and theta > -line_limit1) or (rho < 0 and theta < line_limit1):
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a * rho
					y0 = b * rho
					x1 = int(x0 + 1000*(-b))
					y1 = int(y0 + 1000*(a))
					x2 = int(x0 - 1000*(-b))
					y2 = int(y0 - 1000*(a))

				if assert_line_ylimit(y1) and assert_line_ylimit(y2):
					cv2.line(bgr,(x1, y1),(x2, y2),(0,0,255), 2)
					
					if upper_thresh_paint(y1) and upper_thresh_paint(y2):
						print("FAILURE: Upper limit exceeded. ")
						return bgr
					
					if lower_thresh_paint(y1) and lower_thresh_paint(y2):
						print("FAILURE: Lower limit exceeded. ")
						return bgr
					print("PASSED: Line is within limits")
		return bgr
	else:
		return None

def get_edges(image, sigma=0.33):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), 0)

	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def get_hog(image):
	""" Computes Hisogram of Oriented Gradients. """

	(H, hog_image) = feature.hog(image, orientations=9, pixels_per_cell=(4, 4),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
		visualise=True)
	hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
	hog_image = hog_image.astype("uint8")
	return hog_image

def get_crop(dirn, file, vp, scale):
	""" Crops images """
	img = cv2.imread(os.path.join(dirn, file), 1)
	img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
	
	if vp == VIEWPOINTS[1]:
		x1_p = 0.217
		x2_p = 0.817
		y1_p = 0.645 
		x1 = math.ceil(x1_p * img.shape[1])
		x2 = math.ceil(x2_p * img.shape[1])
		y1 = math.ceil(y1_p * img.shape[0])
		
		bgr_crop = img[y1-1:, x1:x2] 
	
	elif vp == VIEWPOINTS[0]:
		x1_p = 0.191
		x2_p = 0.791
		y1_p = 0.075
		y2_p = 0.430 # y2
		x1 = math.ceil(x1_p * img.shape[1])
		x2 = math.ceil(x2_p * img.shape[1])
		y1 = math.floor(y1_p * img.shape[0])
		y2 = math.ceil(y2_p * img.shape[0])

		bgr_crop = img[y1:y2, x1:x2]
	else:
		print("Unknown viewpoint. Skipping")
		return None

	return bgr_crop
	
	# cv2.imshow("BGR crop", bgr_crop)
	
	# if cv2.waitKey(0) & 0xFF == ord('n'):
	# 	return bgr_crop

	# if cv2.waitKey(0) & 0xFF == ord('q'):
	# 	cv2.destroyAllWindows()
	# 	exit()
	
def post_process(ipath, opath, scale):
	""" Crops images from link-shaft dataset and saves them on specified path. """
	for dir1 in os.listdir(ipath):
		#temp_outpath = os.path.join(opath, dir1)
		#os.mkdir(temp_outpath)
		temp_inpath = os.path.join(ipath, dir1)
	
		for dir2 in os.listdir(temp_inpath):
			#final_path = os.path.join(temp_outpath, dir2)
			#os.mkdir(final_path)
			sub_folder = os.path.join(temp_inpath, dir2)
			with tqdm(total=len(os.listdir(sub_folder))) as prog_bar:
				for file in os.listdir(sub_folder):

					# if dir1 == VIEWPOINTS[1]: # B
					bgr_crop = get_crop(sub_folder, file, dir1, scale)
					cv2.imshow("BGR", bgr_crop)

					hog_crop = get_hog(bgr_crop)
					cv2.imshow("HOG", hog_crop)


						# canny_crop = get_edges(bgr_crop)
						# lines = get_lines(canny_crop, bgr_crop)
						
						# if lines.all() != None:
						# 	cv2.imshow("Lines ", lines)

					if cv2.waitKey(0) & 0xFF == ord('q'):
						return
					prog_bar.update(1)

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True, help="Path to input files")
	ap.add_argument("--outpath", required=True, help="Path to output files")
	ap.add_argument("--scale",   default=1,   help="Resize scale. ", type=float)
	args = ap.parse_args()

	# Process data
	start = t.time() # Program start
	out_dirname = "crop_{}_ver-{}".format(os.path.basename(args.inpath), ((str(dt.datetime.now())).split(' ')[1]).split('.')[0])
	save_to_path = os.path.join(args.outpath, out_dirname)
	#os.mkdir(save_to_path)

	post_process(args.inpath, save_to_path, args.scale)
	
	cv2.destroyAllWindows()
	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()