import cv2
import numpy as np
import glob

def pointExtractor(fname):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(fname)

    for idx, img in enumerate(images):
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

#Insert atleast 20 images
fname = '*.jpg'
objpoints, imgpoints = pointExtractor(fname)

def cameraCalibrator(image):
    imgRes = (image.shape[0], image.shape[1])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgRes, None, None)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    return undistorted

calibrationTest = cv2.imread('camera_cal/calibration1.jpg')
plt.subplot(1,2,1)
plt.imshow(calibrationTest)

undistortedImage = cameraCalibrator(calibrationTest)
plt.subplot(1,2,2)
plt.imshow(undistortedImage)