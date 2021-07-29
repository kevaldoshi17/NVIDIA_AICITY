import os

import cv2
from natsort import natsorted

base = "processed_images/"
base2 = "processed_images2/"

folders = natsorted(os.listdir(base))

for fo in folders:
	files = natsorted(os.listdir(base+fo+'/'))
	if not os.path.isdir(base2+fo):
		os.mkdir(base2+fo)

	for f in files:
		D = cv2.imread(base+fo+'/'+f)
		cv2.imwrite(base2+fo+ "/" +f.split('.')[0] + "_1" + ".jpg",D[:,0:400,:])
		cv2.imwrite(base2+fo+ "/" +f.split('.')[0] + "_2" + ".jpg",D[:,200:600,:])
		cv2.imwrite(base2+fo+ "/" +f.split('.')[0] + "_3" + ".jpg",D[:,400:800,:])

