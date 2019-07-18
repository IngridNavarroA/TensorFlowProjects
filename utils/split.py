
import argparse
import os
import time as t
import itertools
import shutil
import cv2
import random
import errno
			
def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",   required=True, help="Path to input files")
	ap.add_argument("--outpath",  required=True, help="Path to output files")
	ap.add_argument("--test_set", default=0.20,  help="Proportion of images for testing")
	args = ap.parse_args()

	# Process data
	start = t.time() # Program start
	
	# Directory where the training data is
	out_dirname = "train_" + os.path.basename(args.inpath) + "_{}".format(args.test_set)
	out_train = os.path.join(args.outpath, out_dirname)
	if not os.path.exists(out_train):
		os.mkdir(out_train)

	# Directory where the test data is
	out_dirname = "test_" + os.path.basename(args.inpath) + "_{}".format(args.test_set)
	out_test = os.path.join(args.outpath, out_dirname)
	if not os.path.exists(out_test):
		os.mkdir(out_test)
	
	for dirn, _, files in os.walk(args.inpath):
		""" Splits data into training and validation set. """

		if files:
			print("Path is: {}".format(dirn))
			path_split = dirn.split('/')
			viewpoint = path_split[~1]
			clss = path_split[~0]

			test_files = int(len(files) * args.test_set)
			random.shuffle(files)

			temp_out_train = os.path.join(out_train, viewpoint, clss)
			try:
				os.makedirs(temp_out_train)
			except OSError as e:
				if e.errno == errno.EEXIST and os.path.isdir(temp_out_train):
					pass
				else:
					raise

			temp_out_test = os.path.join(out_test, viewpoint, clss)
			try:
				os.makedirs(temp_out_test)
			except OSError as e:
				if e.errno == errno.EEXIST and os.path.isdir(temp_out_test):
					pass
				else:
					raise

			# Test files 
			for i in range(0, test_files):
				image = cv2.imread(dirn+"/"+files[i], 0)
				cv2.imwrite(temp_out_test+"/"+files[i], image)

			# Train files
			for i in range(test_files+1, len(files)):
				image = cv2.imread(dirn+"/"+files[i], 0)
				cv2.imwrite(temp_out_train+"/"+files[i], image)


	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()