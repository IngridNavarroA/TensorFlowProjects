"""
	@author: navarrs
	@date:   October 2nd, 2019
	@brief:  This script is used to download a set of images given a set of keywords
	         and prefixes to perform a Google search. 
	@references:
		- https://github.com/hardikvasa/google-images-download 
"""

from google_images_download import google_images_download
import argparse
import os 

def download( args, google_params ):
	# Method used to download a set of images provided user input.
	response = google_images_download.googleimagesdownload()
	google_params = { "keywords"         : args.keywords,
								    "prefix_keywords"  : args.prefixes,
	                  "limit"            : args.limit,
	              		"output_directory" : args.outpath,
	              		"chromedriver"     : args.chrome,
	              		"format"           : args.format,
		            		"print_urls"       : True }

	paths = response.download( google_params )
	for key in args.keywords.split(','):
		file_path = os.path.join(args.outpath, key+".txt")
		with open(file_path, "a") as f:
			for path in paths[0][args.prefixes + " " + key]:
				f.write(path+"\n")

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("--outpath",  required=True, help="Path to output files")
	ap.add_argument("--keywords", required=True, help="Google search keywords")
	ap.add_argument("--prefixes", help="Prefix keywords to be added to the query")
	ap.add_argument("--limit",    default=20,    help="How many images to download")
	ap.add_argument("--format",    default="jpg", help="Download image format")
	ap.add_argument("--chrome",   help="Path to the chrome driver")
	args = ap.parse_args()

	if not os.path.exists(args.outpath):
		os.mkdir(args.outpath)

	download( args, google_params )
