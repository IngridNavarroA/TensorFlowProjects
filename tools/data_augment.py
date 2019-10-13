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
	ap.add_argument("--inpath",  default="/data/inpath", help="Path to input files")
	ap.add_argument("--samples", default=500, help="Number of augmented samples to generate")

	# Augmentation types
	ap.add_argument("--crop", default=False, help="Apply random crops")
	ap.add_argument("--zoom", default=False, help="Apply random zooms")
	ap.add_argument("--flip_random", default=False, help="Apply random flips")
	ap.add_argument("--flip_x", default=False, help="Apply flips along the horizontal axis")
	ap.add_argument("--brightness", default=False, help="Apply random brightness")
	ap.add_argument("--saturation", default=False, help="Apply random saturation")
	ap.add_argument("--rotate", default=False, help="Apply random rotation")
	ap.add_argument("--distortion", default=False, help="Apply random distortion")
	ap.add_argument("--skew", default=False, help="Apply random skew")


	args = ap.parse_args()

	# Check that input path exists
	assert os.path.exists( args.inpath ), \
		"Input path {} does not exist. ".format( args.inpath )

	p = Augmentor.Pipeline( args.inpath )

	if args.crop:
		p.crop_random( probability=0.5, percentage_area=0.80 )

	if args.zoom:
		p.zoom_random( probability=0.5, percentage_area=0.80 )

	if args.flip_random:
		p.flip_random( probability=0.3 )

	if args.flip_x:
		p.flip_left_right( probability=0.3 )

	if args.brightness:
		p.random_brightness( probability=0.2, min_factor=0.5, max_factor=1.0 )

	if args.saturation:
		p.random_color( probability=0.2, min_factor=0.5, max_factor=1.0 )

	if args.distortion:
		p.random_distortion( probability=0.2, grid_width=4, grid_height=4, magnitude=8 )

	if args.rotate:
		p.rotate( probability=0.3, max_left_rotation=20, max_right_rotation=25 )

	if args.skew:
		p.skew( probability=0.5, magnitude=0.8 )

	p.sample( args.samples, multi_threaded=True )


