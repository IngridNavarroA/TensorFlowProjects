"""
	@author: Ingrid Navarro (navarrs)
	@date:   October 7th, 2019
	@brief:  This script allows to delete or crop images from a given dataset.
					 Provide:
					  --inpath: Full path to the input dataset
	@references:
		- https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
"""
import argparse
import glob 
import cv2
import os 

# Global parameters 
crop_coords = []
cropping = False 

def print_menu():
	# Prints option menu 
	os.system( 'clear' )
	msg = "Key options:\n \
	       \tq -- Quit program\n \
	       \tb -- Get previous image\n \
	       \td -- Delete image\n \
	       \tc -- Crop image\n \
	       \tr -- Reset cropping region\n \
	       \tn -- Show next image\n \
	      "
	print( msg )

def roi_select(event, x, y, flags, param):
	# Given a click event, store coordinates of clicks and draw a bounding 
	# box to indicate the Region of Interest in an image 
	global crop_coords, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		crop_coords = [(x, y)]
		cropping = True

	elif event == cv2.EVENT_LBUTTONUP:
		crop_coords.append((x, y))
		cropping = False

		cv2.rectangle(image, crop_coords[0], crop_coords[1], (0, 0, 255), 2)
		cv2.imshow( "Image", image )

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  default="/data/", help="Path to input files")
	args = ap.parse_args()

	# Check that input path exists
	assert os.path.exists( args.inpath ), \
		"Input path {} does not exist. ".format( args.inpath )

	# Get all files 
	files = [ f for f in glob.glob( args.inpath + "/*.*", recursive=True ) ]
	print_menu()

	# Set mouse callback
	cv2.namedWindow( "Image" )
	cv2.setMouseCallback( "Image", roi_select )
	image = []

	# This text file will keep track of the last image reviewed. 
	index_filename = args.inpath.split('/')[-1] + ".txt"

	# Get the current image index, if the index file exists, get the last index 
	# from the text file, otherwise, the index is 0
	if os.path.exists( index_filename ):
		index_file = open( index_filename, "r" )
		image_index = int( index_file.readline().split()[0] )
		index_file.close()
	else:
		image_index = 0

	while image_index < len( files ):
		file = files[image_index]
		image_index += 1
		
		image_name = file.split('/')[-1]
		image = cv2.imread( file )
		copy_img = image.copy()
		
		forward = True

		while forward:
			cv2.imshow( "Image",  image )
			
			key = cv2.waitKey( 0 )
			if key == ord( 'q' ):
				# Write last index read 
				index_file = open( index_filename, "w" )
				index_file.write( str( image_index ) )
				index_file.close()
				print( "Quitting program" )
				exit( 0 )

			elif key == ord( 'b' ):
				# Get previous image
				if image_index <= 1:
					print( "First image. Cannot go back" )
				else:
					image_index -= 2
					forward = False
					break

			elif key == ord( 'd' ):
				print( "Removing file {}".format(file) )
				os.remove( file )
				break

			elif key == ord( 'c' ):
				# Crop image
				if len(crop_coords) == 2:
					# Get region of interest (roi)
					roi = copy_img[ crop_coords[0][1]:crop_coords[1][1], 
					                crop_coords[0][0]:crop_coords[1][0] ]
					cv2.imshow( "RoI", roi )
					cv2.waitKey(0)

					print( "Replacing crop in {}".format( file ) )

					cv2.imwrite( file, roi )
					cv2.destroyWindow( "RoI" )
					print_menu()
					break
				else:
					print( "Invalid number of reference points" )

			elif key == ord( 'r' ):
				# Reset cropping RoI
				image = copy_img.copy()

			elif key == ord('n'):
				# Break of loop to show next image
				print_menu()
				break

		# When done, remove index file
		if image_index >= len( files ):
			os.remove( index_filename )