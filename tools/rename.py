"""
	@author: navarrs
	@date:   October 2nd, 2019
	@brief: Script renames files to base name.
"""
from datetime import datetime
from skimage import io
import argparse
import glob 
import cv2
import os 

ap = argparse.ArgumentParser()
ap.add_argument("--inpath", default="data", help="Path to data")
ap.add_argument("--name",   default="img",  help="Base name for image files")
ap.add_argument("--format",   default="jpg",  help="Image format")
args = ap.parse_args()

files = [f for f in glob.glob(args.inpath+"/*.*", recursive=True)]

processed_files, correct_files = 1, 1
for file in files:
	filename = "{}_{}_{}.{}".format(args.name, 
		                              correct_files, 
		                              datetime.now().strftime("%d-%m-%Y_%H:%M:%S"), 
		                              args.format)
	try:
		img  = io.imread(file)
		cv2.imwrite(os.path.join(args.inpath, filename), img)
		correct_files += 1
	except:
		print("Image {} could not be read".format(file))

	os.remove(file)
	print("Total files: {}, Correct files: {} Processed files: {}".format(
		     len(files), correct_files, processed_files))
	processed_files += 1