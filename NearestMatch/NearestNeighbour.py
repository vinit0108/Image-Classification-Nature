from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

import os

def image_to_feature_vector(image, size=(32, 32)):
	
	return cv2.resize(image, size).flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


print("describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

imgs = []
labels = []


for (i, imagePath) in enumerate(imagePaths):


	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split("_")[0]


	pixels = image_to_feature_vector(image)


	
	imgs.append(pixels)

	labels.append(label)


	#if i > 0 and i % 100 == 0:
		#print("processed {}/{}".format(i, len(imagePaths)))

imgs = np.array(imgs)

labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	imgs.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
	imgs, labels, test_size=0.25, random_state=10)


print("evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))



