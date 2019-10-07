"""
	@author: navarrs
	@date:   October 2nd, 2019
	@brief: Script converts HDF5 file to npy
"""
from datetime import datetime
from skimage import io
import argparse
import glob 
import h5py
import cv2
import os 

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="Full path to the HDF5 model")
args = ap.parse_args()

with h5py.File(args.file, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

# files = [f for f in glob.glob(args.inpath+"/*.*", recursive=True)]

# processed_files, correct_files = 1, 1
# for file in files:
# 	filename = "{}_{}_{}.{}".format(args.name, 
# 		                              correct_files, 
# 		                              datetime.now().strftime("%d-%m-%Y_%H:%M:%S"), 
# 		                              args.format)
# 	try:
# 		img  = io.imread(file)
# 		cv2.imwrite(os.path.join(args.inpath, filename), img)
# 		correct_files += 1
# 	except:
# 		print("Image {} could not be read".format(file))

# 	os.remove(file)
# 	print("Total files: {}, Correct files: {} Processed files: {}".format(
# 		     len(files), correct_files, processed_files))
# 	processed_files += 1