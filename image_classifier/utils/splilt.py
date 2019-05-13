
import argparse
import os
import time as t
import itertools
import shutil
import cv2

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

def copy_files(opath_train, opath_test, ipath, nfiles, tset):
	test_files = int(nfiles * tset) 
	
	for i in range(len(ipath)):
		x = 0
		path = ipath[i]
		train_path = make_dir(opath_train, os.path.basename(path), False)
		test_path = make_dir(opath_test, os.path.basename(path), False)

		for file in os.listdir(path):
			img = cv2.imread(os.path.join(path, file), 1)
			if x <= test_files: 
				cv2.imwrite(os.path.join(test_path, file), img)
			else:
				cv2.imwrite(os.path.join(train_path, file), img)
			
			x += 1
			if x >= nfiles:
				break
			
def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",   required=True, help="Path to input files")
	ap.add_argument("--outpath",  required=True, help="Path to output files")
	ap.add_argument("--ver",      default=False, help="Keep control of directory versions")
	ap.add_argument("--test_set", default=0.15,  help="Proportion of images for testing")
	args = ap.parse_args()

	# Process data
	start = t.time() # Program start
	
	# Directory where the training data is
	out_dirname = "train_" + os.path.basename(args.inpath) 
	out_train_path = make_dir(args.outpath, out_dirname, args.ver)

	# Directory where the test data is
	out_dirname = "test_" + os.path.basename(args.inpath)
	out_test_path = make_dir(args.outpath, out_dirname, args.ver)
	
	for dir1 in os.listdir(args.inpath):
		""" Splits data into training and validation set. """
		temp_inpath = os.path.join(args.inpath, dir1)
		t1 = os.path.join(temp_inpath, os.listdir(temp_inpath)[0])
		t2 = os.path.join(temp_inpath, os.listdir(temp_inpath)[1])

		# Training set will have n samples per class, where n is the min number of files in either class. 
		num_files = min(len(os.listdir(t1)), len(os.listdir(t2))) 
		copy_files(make_dir(out_train_path, dir1, False), make_dir(out_test_path, dir1, False), [t1, t2], num_files, args.test_set)

	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()