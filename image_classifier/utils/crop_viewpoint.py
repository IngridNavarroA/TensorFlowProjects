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
import shutil
import cv2
import re
import math
import datetime as dt

def crop(dirn, file, vp, scale):
	""" Crops images """
	img = cv2.imread(os.path.join(dirn, file), 1)
	img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
	if not vp:
		x1_p = 0.217
		x2_p = 0.817
		y1_p = 0.645 
		#x1 = int(x1_p * img.shape[1]) - offset
		#x2 = int(x2_p * img.shape[1]) + offset
		#y1 = int(y1_p * img.shape[0]) - offset
		x1 = math.ceil(x1_p * img.shape[1])
		x2 = math.ceil(x2_p * img.shape[1])
		y1 = math.ceil(y1_p * img.shape[0])
		crop_img = img[y1-1:, x1:x2] 
	else:
		x1_p = 0.191
		x2_p = 0.791
		y1_p = 0.075
		y2_p = 0.430 # y2
		#x1 = int(x1_p * img.shape[1]) - offset
		#x2 = int(x2_p * img.shape[1]) + offset
		#y2 = int(y2_p * img.shape[0]) - offset
		x1 = math.ceil(x1_p * img.shape[1])
		x2 = math.ceil(x2_p * img.shape[1])
		y1 = math.floor(y1_p * img.shape[0])
		y2 = math.ceil(y2_p * img.shape[0])
		crop_img = img[y1:y2, x1:x2]
	
	#cv2.imshow("Image", img)
	#cv2.imshow("Cropped", crop_img)
	#if cv2.waitKey(0) & 0xFF == ord('n'):
	#	return crop_img
	#if cv2.waitKey(0) & 0xFF == ord('q'):
	#	cv2.destroyAllWindows()
	#	exit()

	return crop_img

def post_process(ipath, opath, scale):
	""" Crops images from link-shaft dataset and saves them on specified path. """
	for dir1 in os.listdir(ipath):
		temp_outpath = os.path.join(opath, dir1)
		os.mkdir(temp_outpath)
		temp_inpath = os.path.join(ipath, dir1)
		viewpoint = False
		if dir1 == "A":
			viewpoint = True
		for dir2 in os.listdir(temp_inpath):
			final_path = os.path.join(temp_outpath, dir2)
			os.mkdir(final_path)
			sub_folder = os.path.join(temp_inpath, dir2)
			with tqdm(total=len(os.listdir(sub_folder))) as prog_bar:
				for file in os.listdir(sub_folder):
					img = crop(sub_folder, file, viewpoint, scale)
					cv2.imwrite(final_path+"/"+file, img)
					#if cv2.waitKey(0) & 0xFF == ord('q'):
					#	return
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
	os.mkdir(save_to_path)

	post_process(args.inpath, save_to_path, args.scale)
	
	cv2.destroyAllWindows()
	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()