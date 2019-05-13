"""
	@author: Ingrid Navarro
	@date:   Apr 23rd, 2019
	@brief:  Codigo para hacer un GET request del dataset
"""
from tqdm import tqdm
import time as t
import argparse
import os
import datetime as dt
import cv2
import numpy as np
import json
import urllib.request
from multiprocessing import Pool, TimeoutError

VIEWPOINTS = ["A", "B"]
CLASSES    = ["fail", "pass", "unclassified"]
PASSCODES  = [0, 1, 2]
NTHREADS   = 12

base_dir = ''

def get_datetime() -> str:
	""" Gets current date and time. """
	datetime  = ((str(dt.datetime.now())).split(' ')[0]).split('-')[2] + '-' #   Day 
	datetime += ((str(dt.datetime.now())).split(' ')[0]).split('-')[1] + '-' # + Month
	datetime += ((str(dt.datetime.now())).split(' ')[0]).split('-')[0] + '_' # + Year
	datetime += ((str(dt.datetime.now())).split(' ')[1]).split('.')[0]       # + Time
	return datetime

def get_json(url: str) -> str:
	""" Gets JSON data from specified URL. """
	print("Getting JSON object from {}...".format(url))
	req = urllib.request.Request(url)
	with urllib.request.urlopen(req) as res:
		json_object = json.loads(res.read().decode('utf-8'))
		print("[DONE]")
		return json_object

def get_image_urls(json_object: str):
	""" Gets all image URLS. """
	print("Getting image URLS from retrieved JSON...")
	image_urls = []
	for data in json_object:
		image_urls.append(data['imageURL'])

	print("[DONE]")
	return image_urls

def get_image_from_url(url: str, img_name: str):
	""" Converts url to image. """
	req = urllib.request.Request(url)
	with urllib.request.urlopen(req) as res:
		buff = res.read()
		if not buff:
			return None
		image = np.asarray(bytearray(buff), dtype="uint8")
		return cv2.imdecode(image, cv2.IMREAD_COLOR)
		

def create_output_dirs(base_dir: str):
	""" Creates output directories where dataset will be saved. """
	print("Creating output directories...")
	def create_class_dir(vp):
		for i in range(len(CLASSES)):
			class_path = os.path.join(vp, CLASSES[i])  # /data/A/CLASS
			os.mkdir(class_path)

	# Base directory
	os.mkdir(base_dir)

	# Create viewpoint directory
	for i in range(len(VIEWPOINTS)):
		vp_path = os.path.join(base_dir, VIEWPOINTS[i]) # /data/A
		os.mkdir(vp_path)
		# Create class directory
		create_class_dir(vp_path)
	print("[DONE]")

def download_image_from_url(image_url: list):
	# Get image name
	image_name = image_url.split('/')[-1]

	# Get image viewpoint from image name (denoted A or B)
	image_name_split = image_name.split('_')
	viewpoint = image_name_split[0].split('-')[-1]
	if viewpoint in VIEWPOINTS:
		save_to_path = os.path.join(base_dir, viewpoint)
		# If viewpoint exists, we check passcode which is in position 4
		passcode = int(image_name_split[4])
		if passcode in PASSCODES:
			# If passcode exits, update path where image will be written
			save_to_path = os.path.join(save_to_path, CLASSES[passcode], image_name)
			# Get image to download from server
			image = get_image_from_url(image_url, image_name)
			if image is not None:
				cv2.imwrite(save_to_path, image)
			else:
				print("[ERROR] Buffer Empty.")
		else:
			print("[ERROR] Unknown passcode. Skipping Image.")
	else:
		print("[ERROR] Unknown viewpoint. Skipping Image.")

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--outpath", required=True,   help="Path to output files.")
	args = ap.parse_args()

	datetime = get_datetime()
	url = "https://io.xlabhq.com/projects/linkshaft/samples/"
	json_object = get_json(url)

	if json_object:
		start = t.time() # Program start
		# Create output directories to separate data
		global base_dir 
		base_dir = os.path.join(args.outpath, "dataset_{}".format(datetime))
		create_output_dirs(base_dir)

		# Get image URLS
		images_list = get_image_urls(json_object)
		# Download data
		print("Downloading data using {} threads".format(NTHREADS))
		pool = Pool(processes=NTHREADS)
		for _ in tqdm(pool.imap_unordered(download_image_from_url, images_list), total=len(images_list)):
			pass

		print("[DONE] execution Time: ", t.time() - start, "s")
	else:
		print("No data to pull. ")

if __name__ == '__main__':
	main()