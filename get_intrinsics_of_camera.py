import glob
import cv2
import numpy as np

# Define the dimensions of chessboard (number of inner corners)
CHECKERBOARD = (6, 4)

# Stop the iteration when reaching specified criteria (max count = 30, epsilon = 0.001)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector to store 3D points for each checkerboard image
objpoints = []
# Vector to store 2D points for each checkerboard image
imgpoints = []

# Define 3D points for the checkerboard corners (Z coordinate is 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
    -1, 2
)

# Extract path of individual images stored in a given directory
images = glob.glob("calib_data/*.png")

for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

  # If desired number of corners are found, refine them and add to vectors
  if ret:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow("Img", img)
    cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera Matrix :\n", mtx)
print("Distortion Coefficients :\n", dist)
print("Rotation Vectors :\n", rvecs)
print("Translation Vectors :\n", tvecs)

np.save('camera_matrix.npy', mtx)
np.save('distortion_matrix.npy', dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )