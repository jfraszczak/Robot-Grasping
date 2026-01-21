import numpy as np
import cv2 as cv
import glob


# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
def calibrate_camera(
    images_path: str,
    chessboard_size: tuple[int, int],
    square_size: float = 1.0,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    rows, cols = chessboard_size
    rows -= 1
    cols -= 1

    objp = np.zeros((rows * cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(images_path)
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            if verbose:
                # Draw and display the corners
                img_copy: np.ndarray = img.copy()
                cv.drawChessboardCorners(img_copy, (rows, cols), corners2, ret)
                cv.imshow('img', img_copy)
                cv.waitKey(0)

    if verbose:
        cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist
