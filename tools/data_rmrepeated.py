"""
	@author: navarrs
	@date:   October 2nd, 2019
	@brief:  This script removes repeated files obtained using the 
					 google_download.py script.
	@references:
		- https://github.com/hardikvasa/google-images-download
"""
import glob 
import argparse
import os 

def rm_repeated(files):
	# This method deletes repeated images based on the name of the image. This 
	# method works when using multiple times the google-images-download package
	# because the method will download same image several times. 
	print("Reading {} files".format( len( files ) ) )
	files = [ f.split('.') for f in files ]
	files.sort(key=lambda x: x[1])
	current_element = files[0][1]
	counter = 0
	for i in range(1, len(files)):
		if current_element == files[i][1]:
			file = '.'.join(files[i])
			os.remove(file)
			counter += 1
		else:
			current_element = files[i][1]
	print("Removed {} files".format(counter))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath", default="data", help="Path to data")
	ap.add_argument("--format", default="jpg,png", help="Image format(s)")
	args = ap.parse_args()

	img_formats = args.format.split(',')
	for img_format in img_formats:
		files = [f for f in glob.glob(args.inpath+"/*.{}".format(img_format), recursive=True)]
		rm_repeated(files)
