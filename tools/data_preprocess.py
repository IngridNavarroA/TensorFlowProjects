"""
	@author: navarrs
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
bbox = []
cropping = False 

def print_menu( f, n ):
	# Prints option menu 
	os.system( 'clear' )
	msg = "Key options:\n \
	       \tq -- Quit program\n \
	       \ts -- Save progress\n \
	       \tb -- Get previous image\n \
	       \td -- Delete image\n \
	       \tc -- Crop image\n \
	       \tr -- Reset cropping region\n \
	       \tn -- Show next image\n\n \
	       Processed images {}/{} \
	      ".format( f, n )
	print( msg )

def check_coord(x, y, w, h):
	if x >= w:
		x = w
	elif x < 0:
		x = 0

	if y >= h:
		y = h
	elif y < 0:
		y = 0

	return x, y

def roi_select(event, x, y, flags, param):
	# Given a click event, store coordinates of clicks and draw a bounding 
	# box to indicate the Region of Interest in an image 
	global bbox, cropping

	height, width = image.shape[:2]

	if event == cv2.EVENT_LBUTTONDOWN:
		x, y = check_coord(x, y, width, height)
		bbox = [(x, y)]
		cropping = True

	elif event == cv2.EVENT_LBUTTONUP:
		x, y = check_coord(x, y, width, height)
		bbox.append((x, y))
		cropping = False

		# Swap coordinates if they were grabbed from bottom to top
		if bbox[1][1] < bbox[0][1]:
			bbox[0], bbox[1] = bbox[1], bbox[0]

		cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 2)
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
	nfiles = len( files )
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

	print_menu( image_index, len( files ) )
	deleted_images = 0
	while image_index < nfiles:
		file = files[image_index]
		image_index += 1
		
		image_name = file.split('/')[-1]
		image = cv2.imread( file )
		image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
		copy_img = image.copy()
		
		forward = True

		while forward:
			cv2.imshow( "Image",  image )
			
			key = cv2.waitKey( 0 )
			if key == ord( 'q' ):
				# Write last index read 
				index_file = open( index_filename, "w" )
				index_file.write( str( image_index - deleted_images ) )
				index_file.close()
				print( "Quitting program" )
				exit( 0 )

			elif key == ord( 's' ):
				index_file = open( index_filename, "w" )
				index_file.write( str( image_index - deleted_images ) )
				index_file.close()
				print( "Saving progress" )

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
				nfiles -= 1
				deleted_images += 1
				break

			elif key == ord( 'c' ):
				# Crop image
				if len(bbox) == 2:
					# Get region of interest (roi)
					roi = copy_img[ bbox[0][1]:bbox[1][1], 
					                bbox[0][0]:bbox[1][0] ]
					cv2.imshow( "RoI", roi )
					cv2.waitKey(0)

					print( "Replacing crop in {}".format( file ) )

					cv2.imwrite( file, roi )
					cv2.destroyWindow( "RoI" )
					print_menu( image_index, nfiles )
					break
				else:
					print( "Invalid number of reference points" )

			elif key == ord( 'r' ):
				# Reset cropping RoI
				image = copy_img.copy()

			elif key == ord('n'):
				# Break of loop to show next image
				print_menu( image_index, nfiles )
				break

		# When done, remove index file
		if image_index >= nfiles:
			os.remove( index_filename )