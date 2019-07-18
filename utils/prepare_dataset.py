"""
	@author: Ingrid Navarro
	@date: Feb 12th, 2019
	@brief: Helper method to create text files for training and testing data with Alexnet
"""
from sklearn.utils import shuffle
from statistics import mode

import time as t
import itertools
import argparse
import sys
import os
import shutil

def to_text_file(str_list, file_path):
	""" Creates dataset text file """
	with open(file_path, "a") as f:
		for i in range(len(str_list)):
			f.write(str_list[i] + "\n")
		f.close()

def prep_data(ipath, opath, val):
	""" Crops images from link-shaft dataset and saves them on specified path. """
	cls = 0 
	dataset = dict()
	for dir1 in os.listdir(ipath):
		dataset[cls] = []
		temp_inpath = os.path.join(ipath, dir1)
		for file in os.listdir(temp_inpath):
			image = "{} {}".format(os.path.join(temp_inpath, file), cls)
			dataset[cls].append(image)
		cls += 1 
	return dataset

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True, help="Path to input files")
	ap.add_argument("--outpath", required=True, help="Path to output files")
	ap.add_argument("--testset", default=0.20,  help="Proportion to split dataset into train / val")
	args = ap.parse_args()

	# Process data
	start = t.time() # Program start
	dataset = prep_data(args.inpath, args.outpath, args.testset)

	val_set = []
	train_set = []
	for cls, data in dataset.items():
		limit = int(len(data) * args.testset)
		val_set += data[0:limit]
		train_set += data[limit+1:]
	
	val_set = shuffle(val_set)
	train_set = shuffle(train_set)

	to_text_file(val_set, os.path.join(args.outpath, "val.txt"))
	to_text_file(train_set, os.path.join(args.outpath, "train.txt"))

	print("[DONE] execution Time: ", t.time() - start, "s")

if __name__ == '__main__':
	main()