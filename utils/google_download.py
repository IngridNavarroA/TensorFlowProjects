from google_images_download import google_images_download
import argparse
import os 

ap = argparse.ArgumentParser()
ap.add_argument("--outpath",  required=True, help="Path to output files")
ap.add_argument("--keywords", required=True, help="Google search keywords")
ap.add_argument("--prefixes", help="Prefix keywords to be added to the query")
ap.add_argument("--limit",    default=20,    help="How many images to download")
args = ap.parse_args()

if not os.path.exists(args.outpath):
	os.mkdir(args.outpath)

response = google_images_download.googleimagesdownload()

arguments = { "keywords"         : args.keywords,
							"prefix_keywords"  : args.prefixes,
              "limit"            : args.limit,
              "output_directory" : args.outpath,
              "format"           : "png",
	            "print_urls"       : True 
	          }
paths = response.download(arguments)

for key in args.keywords.split(','):
	file_path = os.path.join(args.outpath, key+".txt")
	with open(file_path, "a") as f:
		for path in paths[0][key]:
			f.write(path+"\n")
