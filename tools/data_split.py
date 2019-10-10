"""
	@author: navarrs
	@date:   September 27th, 2019
	@brief:  Script to split a dataset into train / test sets. 
"""
import argparse
import os
import cv2
import random
import glob
		
def save_files( path, files ):
	for file in files:
		file_name = file.split("/")[-1]
		img = cv2.imread( file )
		cv2.imwrite( os.path.join( path, file_name ), img )

ap = argparse.ArgumentParser()
ap.add_argument("--inpath",   required=True, help="Path to input files")
ap.add_argument("--outdir", default="split_data", help="Directory where the data will be split and saved")
ap.add_argument("--outpath",  required=True, help="Path to output files")
ap.add_argument("--test_set", default=0.15,  help="Proportion of images for testing")
ap.add_argument("--img_format", default="png", help="Image format [ png | jpg ]")
args = ap.parse_args()

# Create base output directory and train and test sub-directories
if not os.path.exists( args.outpath ):
	os.makedirs( args.outpath )

train_path = os.path.join( args.outpath, "train_data", args.outdir )
if not os.path.exists( train_path ):
	os.makedirs( train_path )

test_path = os.path.join( args.outpath, "test_data", args.outdir )
if not os.path.exists( test_path ):
	os.makedirs( test_path )

files = [ f for f in glob.glob( args.inpath + "**/*."+args.img_format ) ]
random.shuffle( files )
num_files = len( files )
test_files = int( num_files * args.test_set )

print( "Total files {} - Train files {} / Test files {}".format( 
	num_files, num_files - test_files, test_files) )

train_files = files[test_files:]
test_files  = files[:test_files]
save_files( train_path, train_files )
save_files( test_path,  test_files  )