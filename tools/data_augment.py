"""
	@author: navarrs
	@date:   October 7th, 2019
	@brief:  This script allows to delete or crop images from a given dataset.
					 Provide:
					  --inpath: Full path to the input dataset
	@references:
		- https://github.com/mdbloice/Augmentor/blob/master/Augmentor
"""
import argparse
import Augmentor
import os

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  default="/data/", help="Path to input files")
	ap.add_argument("--samples", default=500, help="Number of augmented samples to generate")
	args = ap.parse_args()

	# Check that input path exists
	assert os.path.exists( args.inpath ), \
		"Input path {} does not exist. ".format( args.inpath )

	p = Augmentor.Pipeline( args.inpath )

	p.zoom_random(probability=0.5, percentage_area=0.75)
	p.flip_left_right(probability=0.5)
	p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
	p.crop_random(probability=0.5, percentage_area=0.5)

	p.sample( args.samples, multi_threaded=True )


