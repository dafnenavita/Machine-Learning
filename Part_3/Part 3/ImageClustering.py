import numpy as np
import os
import cv2
import sys

#img1 = cv2.imread('image1.jpg')
img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])
img3 = cv2.imread(sys.argv[3])

path = 'clusteredImages'
os.mkdir(path)
images = [img1, img2, img3]
image_id = 1
for img in images:
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    cv2.imwrite(os.path.join(path, str(image_id)+'.jpg'), res2)
    image_id += 1
    cv2.waitKey(0)





