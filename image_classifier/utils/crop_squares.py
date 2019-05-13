"""
	@author: Ingrid Navarro
	@date: Feb 26th, 2019
	@brief: Codigo para penerar ejemplos cuadrados
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
import math

def make_dir(path, dir_name, ver_ctrl):
	""" Create output directory. Provide path, directory name, and version control. 
		The method will create a new version of the directory if it already exists, 
		unless ver_ctrl is unset.  
	"""
	if ver_ctrl:
		ver = 1
		new_dir = dir_name + str(ver)
		full_path = os.path.join(path, new_dir)
		while os.path.isdir(full_path):
			print("[INFO] Directory {} exists.".format(full_path))
			ver += 1
			new_dir = dir_name + str(ver)
			full_path = os.path.join(path, new_dir)
	else:
		full_path = os.path.join(path, dir_name)
		print(full_path)

		if os.path.isdir(full_path):
			shutil.rmtree(full_path)
		
	print("[INFO] Creating directory {}".format(full_path))
	os.makedirs(full_path)
	return full_path

def crop(opath, dirn, file, sub_img_w=128):
	""" Crops images """
	img = cv2.imread(os.path.join(dirn, file), 0)

	for i in range(0, img.shape[0], sub_img_w):
		for j in range(0, img.shape[1], sub_img_w):
			sub_img = img[i:i+sub_img_w, j:j+sub_img_w]
			cv2.imshow("square1", sub_img)
			cv2.imwrite(opath+"/"+"sub{}-{}_".format(i, j)+file, sub_img)
		
	#sub_img1 = img[:, :sub_img_w-1]
	#sub_img2 = img[:, sub_img_w:2*sub_img_w-1]
	#sub_img3 = img[:, 2*sub_img_w:]

	#cv2.imshow("square1", sub_img1)
	#cv2.imshow("square2", sub_img2)
	#cv2.imshow("square3", sub_img3)

	#if cv2.waitKey(0) & 0xFF == ord('n'):
	#	return sub_img1, sub_img2, sub_img3
	#if cv2.waitKey(0) & 0xFF == ord('q'):
	#	cv2.destroyAllWindows()
	#	exit()

def post_process(ipath, opath):
	""" Crops images from link-shaft dataset and saves them on specified path. """
	for dir1 in os.listdir(ipath):
		temp_outpath = make_dir(opath, dir1, False)
		temp_inpath = os.path.join(ipath, dir1)
		
		for dir2 in os.listdir(temp_inpath):
			final_path = make_dir(temp_outpath, dir2, False)
			sub_folder = os.path.join(temp_inpath, dir2)
			with tqdm(total=len(os.listdir(sub_folder))) as prog_bar:
				for file in os.listdir(sub_folder):
					crop(final_path, sub_folder, file)
					# cv2.imwrite(final_path+"/"+"sub1_"+file, img1)
					# cv2.imwrite(final_path+"/"+"sub2_"+file, img2)
					# cv2.imwrite(final_path+"/"+"sub3_"+file, img3)

					prog_bar.update(1)

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True, help="Path to input files")
	ap.add_argument("--outpath", required=True, help="Path to output files")
	ap.add_argument("--ver",     default=False, help="Keep control of directory versions")
	args = ap.parse_args()

	# Process data
	start = t.time() # Program start
	out_path = make_dir(args.outpath, "sq_" + os.path.basename(args.inpath), args.ver)
	post_process(args.inpath, out_path)
	
	cv2.destroyAllWindows()
	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()