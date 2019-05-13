"""
	author: Ingrid Navarro
	date: Dec 17th, 2018
	brief: Set of utils to make operations on images. 
"""
from tqdm import tqdm
from statistics import mode
import numpy as np
import time as t
import itertools
import argparse
import sys
import os
import shutil
import cv2
import re

def make_dir(path, dir_name, ver_ctrl):
	"""
		Create output directory. Provide path, directory name, and version control. 
		The method will create a new version of the directory if it already exists, 
		unless ver_ctrl is unset.  
	"""
	if ver_ctrl:
		ver = 1
		new_dir = dir_name + str(ver)
		full_path = os.path.join(path, new_dir)
		while os.path.isdir(full_path):
			print "[INFO] Directory {} exists.".format(full_path)
			ver += 1
			new_dir = dir_name + str(ver)
			full_path = os.path.join(path, new_dir)
	else:
		full_path = os.path.join(path, dir_name)
		print full_path

		if os.path.isdir(full_path):
			shutil.rmtree(full_path)
		
	print "[INFO] Creating directory {}".format(full_path)
	os.makedirs(full_path)
	return full_path

def resize(in_path, out_path, scale, debug):
	"""
		Resize images from specified directory to specified scale. 
	"""
	num_files = len(os.listdir(in_path))	
	print "\n[INFO] Processing {:} files from: {}".format(num_files, in_path)
	with tqdm(total=num_files) as prog_bar:
		for file in os.listdir(in_path):
			if not file.startswith("."):
				# Read image
				org_img = cv2.imread(os.path.join(in_path, file), 1)#, cv2.CV_LOAD_IMAGE_COLOR)

				# Resize image
				new_w, new_h = org_img.shape[1] * scale, org_img.shape[0] * scale
				new_img = cv2.resize(org_img, (int(new_w), int(new_h)))

				if debug:
					while not cv2.waitKey(1) & 0xFF == ord ('q'):
						cv2.imshow("original", org_img)
						cv2.imshow("resized", new_img)
				else:
					# Save image
					file_name, ext = os.path.splitext(file)
					file_name += "_{}".format(scale) + ".jpg"
					cv2.imwrite(out_path + "/" + file_name, new_img)
				
				prog_bar.update(1)

def draw_roi(img, p):
	"""
		Draws the Region of Interest that we want to warp
	"""
	cv2.circle(img, (p[0][0], p[0][1]), 5, (0, 0, 255), -1)
	cv2.circle(img, (p[1][0], p[1][1]), 5, (0, 0, 255), -1)
	cv2.circle(img, (p[2][0], p[2][1]), 5, (0, 0, 255), -1)
	cv2.circle(img, (p[3][0], p[3][1]), 5, (0, 0, 255), -1)
	
	cv2.line(img, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0, 255, 0), 2)
	cv2.line(img, (p[0][0], p[0][1]), (p[3][0], p[3][1]), (0, 255, 0), 2)
	cv2.line(img, (p[2][0], p[2][1]), (p[1][0], p[1][1]), (0, 255, 0), 2)
	cv2.line(img, (p[2][0], p[2][1]), (p[3][0], p[3][1]), (0, 255, 0), 2)

def homography(org_img, points):
	"""
		Performs perspective transform 
	"""
	(tl, tr, br, bl) = points

	# Compute width of warped image
	top_width = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
	bottom_width = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
	width = max(int(top_width), int(bottom_width))

	# Height of warped image 
	height = br[0]
	
	print "[INFO] size of warped image ({}, {})".format(height, width)

	out_img = np.array([[0, 0], 
		                [width-1, 0],
		                [width-1, height-1], 
		                [0, height-1]], 
		                dtype="float32")

	# Compute affine transform
	M = cv2.getPerspectiveTransform(np.asarray(points), out_img)
	warped_img = cv2.warpPerspective(org_img, M, (width, height))

	return warped_img

def show_warp(org_img, points, warped, edges):
	"""
		Shows perspective transform
	"""
	img = org_img.copy()
	draw_roi(img, points)
	while not cv2.waitKey(1) & 0xFF == ord ('n'):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print "[INFO] Exiting program..."
			exit()

		cv2.imshow('image', img)
		cv2.imshow('warped', warped)
		cv2.imshow('canny', edges)

def canny(img, sigma=0.33):
	v = np.median(img)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return cv2.Canny(img, lower, upper)

def warp(in_path, out_path, folder, debug):
	"""
		Warp image perspective
	"""
	num_files = len(os.listdir(in_path))	
	print "\n[INFO] Processing {:} files from: {}".format(num_files, in_path)
	with tqdm(total=num_files) as prog_bar:
		for file in os.listdir(in_path):
			if not file.startswith("."):
				# Read image
				org_img = cv2.imread(os.path.join(in_path, file), 1)#, cv2.CV_LOAD_IMAGE_COLOR)
				img = org_img.copy()

				# General offsets
				# TODO: calibrate offsets on fixed camera
				H, W = org_img.shape[0], org_img.shape[1]
				top_h    = int(0.15 * H) 
				bottom_h = int(0.75 * H)

				run = False

				# Get coordinates
				if folder == "abajo":
					# Homography offsets from botton viewpoint
					top_left_w  = int(0.10 * W)
					top_right_w = int(0.80 * W)
					bottom_left_w  = int(0.05 * W)
					bottom_right_w = int(0.95 * W)

					points = np.zeros((4, 2), dtype="float32")
					points[0] = [top_left_w, top_h] 
					points[1] = [top_right_w, top_h]
					points[2] = [bottom_right_w, bottom_h]
					points[3] = [bottom_left_w, bottom_h]

					warped = homography(org_img, points)
					edges = cv2.Canny(warped, 100, 200)

					if debug:
						show_warp(org_img, points, warped, edges)
					else:
						cv2.imwrite(out_path + "/" + file, warped)

				elif folder == "arriba":
					# Homography offsets from top viewpoint 
					top_left_w  = int(0.15 * W)
					top_right_w = int(0.85 * W)
					bottom_left_w  = int(0.25 * W)
					bottom_right_w = int(0.75 * W)

					points = np.zeros((4, 2), dtype="float32")
					points[0] = [top_left_w, top_h] 
					points[1] = [top_right_w, top_h]
					points[2] = [bottom_right_w, bottom_h]
					points[3] = [bottom_left_w, bottom_h]

					warped = homography(org_img, points)
					gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
					blurred = cv2.GaussianBlur(gray, (3, 3), 0)
					edges = canny(blurred)

					if debug:
						show_warp(org_img, points, warped, edges)
					else:
						warped = homography(org_img, points)

						cv2.imwrite(out_path + "/" + file, warped)

				else:
					print "[Error] unknown folder"
					exit()
				
				prog_bar.update(1)

def find_word(w):
	"""
		Search word w in string
	"""
	return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def split(in_path, out_path, ver_ctrl):
	"""
		Split dataset into defective and non-defective images
	"""
	num_files = len(os.listdir(in_path))	
	print "\n[INFO] Processing {:} files from: {}".format(num_files, in_path)

	bad = make_dir(out_path, "bad", ver_ctrl)
	good = make_dir(out_path, "good", ver_ctrl)

	with tqdm(total=num_files) as prog_bar:
	 	for file in os.listdir(in_path):

	 		image = cv2.imread(in_path + file)
	 		temp_name = file.replace('_', " ").replace(".jpg", " ")
	 		if not find_word('pass')(temp_name):
	 			cv2.imwrite(bad + "/" + file, image)
	 		else:
	 			cv2.imwrite(good + "/" + file, image)

	 		prog_bar.update(1)

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True,     help="Path to input files")
	ap.add_argument("--outpath", required=True,     help="Path to output files")
	ap.add_argument("--conv",    required=True,     help="Operation to perform: [ resize | warp ]")
	ap.add_argument("--scale",   default=2.0,       help="Resize Scale", type=float)
	ap.add_argument("--debug",   default=False,     help="Specify if debugging")
	ap.add_argument("--ver",     default=False,     help="Keep control of directory versions")
	args = vars(ap.parse_args())

	in_path   = args["inpath"]
	out_path  = args["outpath"]
	conv_type = args["conv"]
	ver_ctrl  = args["ver"] 
	debug     = args["debug"]
	scale     = args["scale"]

	# Process data
	start = t.time() # Program start
	if conv_type == "split":
		split(in_path, out_path, ver_ctrl)
	else:
	 	for folder in os.listdir(in_path):
	 		if not folder.startswith('.'):
		 		inpath = os.path.join(in_path, folder)
		 		outpath = os.path.join(out_path, in_path.split("/")[-1])
		 		out_dir = folder + '_' + conv_type

				if conv_type == "resize": 
					whole, frac = int(scale), int(scale * 100)
					out_dir += str(whole) + '_' + str(frac)
					# print "input {}, output {}".format(inpath, outpath)
					resize(inpath, make_dir(outpath, out_dir, bool(ver_ctrl)), scale, bool(debug))
				elif conv_type == "warp":
					warp(inpath, make_dir(outpath, out_dir, bool(ver_ctrl)), folder, bool(debug))
				else:
					print "[Error] invalid argument"
					exit()

	print "[DONE] execution Time: ", t.time() - start, "s"

if __name__ == '__main__':
	main()