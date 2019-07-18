

import cv2
import os

bpath = "./data/dataset_17-06-2019_11:54:35/B"
ipath = os.path.join(bpath, "unclassified")

def mvfile(filename, im_path, status):
	if status == 0:
		# fail
		 out_path = os.path.join(bpath, "fail", filename)
	elif status == 1:
		# pass
		out_path = os.path.join(bpath, "pass", filename)
	os.rename(im_path, out_path)

for filename in os.listdir(ipath):
	image_path = os.path.join(ipath, filename)
	image = cv2.imread(image_path)

	cv2.imshow("Image", image)

	key = cv2.waitKey(0) & 0xFF
	if key == ord('q'):
		cv2.destroyAllWindows()
		exit()

	if key == ord('0') or key == ord('1'):
		status = key - 48 
		new_filename = filename.split('_')
		new_filename[~1] = str(status)
		new_filename = '_'.join(new_filename)
		mvfile(new_filename, image_path, status)

	if key == ord('2'):
		print("Deleting file {}...".format(filename))
		os.remove(image_path)
