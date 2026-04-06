# OpenCV camera intrinsic parameters.
# Maps directly to cv2.calibrateCamera output.

float64 fx          # focal length x (pixels)
float64 fy          # focal length y (pixels)
float64 cx          # principal point x (pixels)
float64 cy          # principal point y (pixels)
float64[5] dist     # distortion coefficients [k1, k2, p1, p2, k3]
uint32 width        # image width (pixels)
uint32 height       # image height (pixels)
