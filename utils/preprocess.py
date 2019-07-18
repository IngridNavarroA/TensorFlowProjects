"""
	@author: Ingrid Navarro Anaya
	@brief:  Dataset preprocessing. 
			 Crops images and computes local variance to find 
			 paint defects. 
	@date:   June 3rd, 2019
"""

import os 
import cv2 
import argparse
import time as t
import numpy as np
import datetime as dt
from tqdm import tqdm
from shutil import rmtree

def get_crop(image, crop_type, viewpoint):
	""" Crops input image into squares or viewpoint-based."""
	if crop_type == "v":
		if viewpoint == "A":
			x1_p = 0.191
			x2_p = 0.791
			y1_p = 0.075
			y2_p = 0.430
		elif viewpoint == "B":
			x1_p = 0.217
			x2_p = 0.817
			y1_p = 0.645
			y2_p = 1.000 
	elif crop_type == "s":
		if viewpoint == "A":
			x1_p = 0.226
			x2_p = 0.750
			y1_p = 0.069
			y2_p = 1.000
		elif viewpoint == "B":
			x1_p = 0.250
			x2_p = 0.776
			y1_p = 0.000
			y2_p = 0.930

	x1 = np.ceil(x1_p * image.shape[1]).astype('int')
	x2 = np.ceil(x2_p * image.shape[1]).astype('int')
	y1 = np.ceil(y1_p * image.shape[0]).astype('int')
	y2 = np.ceil(y2_p * image.shape[0]).astype('int')
	return image[y1:y2, x1:x2] # Sliced image

def get_blur(image, blur_type):
	""" Blur image using Gaussian or Bilateral filters. """
	if blur_type == "g":
		kernel_size = 5
		sigmax, sigmay = 0, 0
		return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmax, sigmay)
	elif blur_type == "b":
		diam_n = 9
		sigma_color, sigma_space = 75, 75
		return cv2.bilateralFilter(image, diam_n, sigma_color, sigma_space)

def get_local_VAR(image):
	""" Computes local variance on input image. """
	kernel_size = 5
	kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
	image_mean = cv2.filter2D(image, -1, kernel)
	return (image - image_mean) ** 2

def pre_process(args, odirn):
	""" Pre-processes images for defect detection. """

	for dirn, _, files in os.walk(args.inpath):
		print("Found directory: ", dirn)

		# If there are files to process, get the viewpoint to crop
		if files:
			path_split = dirn.split('/')
			viewpoint = path_split[~1]
			clss = path_split[~0]
			temp_opath = os.path.join(args.outpath, odirn, viewpoint, clss)

			try:
				os.makedirs(temp_opath)
			except OSError as e:
				if e.errno == errno.EEXIST and os.path.isdir(temp_opath):
					pass
				else:
					raise

			with tqdm(total=len(files)) as prog_bar:
				for f in files:
					fpath = os.path.join(dirn, f)
					img  = cv2.imread(fpath)

					if args.gray == "y":
						img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					
					crop_img  = get_crop(img, args.crop, viewpoint)
					if args.filter == "n":
						cv2.imwrite(temp_opath+"/"+f, crop_img)
					else:
						blur_img  = get_blur(crop_img, args.filter)
						local_VAR = get_local_VAR(blur_img)
						cv2.imwrite(temp_opath+"/"+f, local_VAR)

					prog_bar.update(1)

					if args.debug:
						cv2.imshow("Image", gray_img)
						cv2.imshow("Crop {}".format(args.crop), crop_img)
						cv2.imshow("Blur {}".format(args.filter), blur_img)
						cv2.imshow("Local VAR", local_VAR)

						key = cv2.waitKey(0) & 0xFF
						if key == ord('q'):
							rmtree(os.path.join(args.outpath, odirn))
							cv2.destroyAllWindows()
							return
						if key == ord('s'):
							break 
						if key == ord('n'):
							continue

def str_to_bool(v: str) -> bool:
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath", 	required=True, 	help="Path to input files")
	ap.add_argument("--outpath", 	default="data", help="Path to save files")
	ap.add_argument("--crop", 		default="v", 	help="Type of crop [viewpoint (v) | square (s)]")
	ap.add_argument("--filter",  	default="b",   	help="Type of filtering [Gauss (g) | Bilateral (b) | No filter (n)]")
	ap.add_argument("--debug", 		default="true", help="Visualize preprocessing")
	ap.add_argument("--gray", 		default="y",    help="COnvert to grayscale? [yes (y) | no (n)]")
	args = ap.parse_args()
 	
	# Assert input arguments
	args.debug = str_to_bool(args.debug)
	assert args.crop == "v" or args.crop == "s", \
		"Crop type not supported. Only: v | s"

	assert args.filter == "b" or args.filter == "g" or args.filter == "n", \
		"Filtering type not supported. Only: g | b"

	assert os.path.isdir(args.inpath), \
		"Directory does not exist."

	start = t.time()
	odirn = "feat-{}-{}-{}_{}_v-{}".format(args.crop, args.filter, args.gray, os.path.basename(args.inpath), ((str(dt.datetime.now())).split(' ')[1]).split('.')[0])
	pre_process(args, odirn)
	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()
