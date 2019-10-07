import argparse
import glob 
import cv2
import os 

def print_menu():
	os.system( 'clear' )
	msg = "Key options:\n \
	       \tq -- Quit program\n \
	       \td -- Delete image\n \
	       \tc -- Crop image\n \
	       \tn -- Show next image\n \
	      "
	print( msg )

ap = argparse.ArgumentParser()
ap.add_argument("--inpath",  default="/data/", help="Path to input files")
ap.add_argument("--outpath", default="/data/", help="Path to output files")
args = ap.parse_args()

assert os.path.exists( args.inpath ), \
	print( "Input path {} does not exist. ".format( args.inpath ) )

if not os.path.exists( args.outpath ):
		os.makedirs( args.outpath )

files = [ f for f in glob.glob(args.inpath+"/*.*", recursive=True) ]
print_menu()

for file in files:
	org_img = cv2.imread( file )
	copy_img = org_img
	(h, w, _ ) = copy_img.shape
	cv2.imshow( "Original Image",  copy_img )
	
	key = cv2.waitKey( 0 )
	if key == ord( 'q' ):
		print( "Quitting program" )
		exit( 0 )
	elif key == ord( 'd' ):
		print("Removing file {}".format(file) )
		os.remove( file )
	elif key == ord( 'c' ):
		print("Set crop points ")
		pass

	print_menu()

