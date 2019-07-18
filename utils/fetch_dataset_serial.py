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

VIEWPOINTS = ["A", "B"]
CLASSES    = ["fail", "pass", "unclassified"]
PASSCODES  = [0, 1, 2]

# TODO: fix last session pull 

def get_datetime() -> str:
	""" Gets current date and time. """
	datetime  = ((str(dt.datetime.now())).split(' ')[0]).split('-')[2] + '-' #   Day 
	datetime += ((str(dt.datetime.now())).split(' ')[0]).split('-')[1] + '-' # + Month
	datetime += ((str(dt.datetime.now())).split(' ')[0]).split('-')[0] + '_' # + Year
	datetime += ((str(dt.datetime.now())).split(' ')[1]).split('.')[0]       # + Time
	return datetime

def get_json(url: str, session: bool) -> str:
	""" Gets JSON data from specified URL. """
	req = urllib.request.Request(url)
	with urllib.request.urlopen(req) as res:
		json_object = json.loads(res.read().decode('utf-8'))
		if session:
			return json_object
		json_object = {"1" : json_object }
		return json_object

def get_image_from_url(url: str, img_name: str):
	""" Converts url to image. """
	req = urllib.request.Request(url)
	with urllib.request.urlopen(req) as res:
		image = np.asarray(bytearray(res.read()), dtype="uint8")
	return cv2.imdecode(image, cv2.IMREAD_COLOR)

def create_output_dirs(base_dir: str):
	""" Creates output directories where dataset will be saved. """
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

def log_session(date, sess_id):
	""" Logs session ID of last session. """
	log_msg = "Date: {} ID: {}\n".format(date, sess_id)
	with open("utils/session.txt", "a") as f:
		f.write(log_msg)
		f.close()

def download_data(json_object: list, base_dir: str, last_session: int):
	for key, value in json_object.items():
		with tqdm(total=len(value)) as prog_bar:
			for data in value:
				image_url  = data['imageURL']
				current_session = int(data['sessionId'])
				if current_session > last_session:
					last_session = current_session
					log_session(get_datetime(), str(last_session))

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
						cv2.imwrite(save_to_path, image)
					else:
						print("[ERROR] Unknown passcode. Skipping Image.")
				else:
					print("[ERROR] Unknown viewpoint. Skipping Image.")

				prog_bar.update(1)

def get_session_id():
	""" Gets ID of last pulled JSON """
	try:
		with open("utils/session.txt", "r") as f:
			content = f.readlines()
			f.close()
	except IOError:
		print("Error. Creating file")
		return "1"

	content = [x.strip() for x in content]
	if not content: # Pull everything
		return 0
	else:
		return int(content[-1].split(' ')[-1]) # Last pull 

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
	ap.add_argument("--outpath", required=True,   help="Path to output files.")
	ap.add_argument("--ver",     default=False,   help="Keep control of directory versions.")
	ap.add_argument("--session", default=False,   help="Download data from last session.")
	args = ap.parse_args()

	sess = str_to_bool(args.session)
	datetime = get_datetime()
	# Get data from last session
	url = "https://io.xlabhq.com/projects/linkshaft/samples/"
	last_session = get_session_id()
	if sess:
		if last_session != 0:
			url += "from/?sessionId=" + str(last_session)
		else: 
			sess = False

	json_object = get_json(url, sess)

	if json_object:
		start = t.time() # Program start
		# Create output directories to separate data
		base_dir = os.path.join(args.outpath, "dataset_sess-{}_{}".format(last_session, datetime))
		create_output_dirs(base_dir)

		# Download data
		download_data(json_object, base_dir, last_session)
		print("[DONE] execution Time: ", t.time() - start, "s")
	else:
		print("No data to pull. ")

if __name__ == '__main__':
	main()