import cv2
import numpy as np
import imutils
import transform

# read all the 2 pages
full = cv2.imread('scanned.jpg')

# reduce dimensions
ratio = full.shape[0] / 500.0
orig = full.copy()
full = imutils.resize(full, height=500)

# grayscale it
gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)

# apply a low pass filter and threahold it
gray = cv2.GaussianBlur(gray, (3, 15), 0)

# threshold the image and convert it to binary
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# find the edges
edge = cv2.Canny(gray, 75, 200)

# fil the space
# edge1 = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, (25,25), iterations = 1)
edge = cv2.dilate(edge, (101, 101))
rows, cols = edge.shape

# new = cv2.line(full, (cols/2, 0), (cols/2, rows), (0,255,0), 1)
pts = np.array([[0.4 * cols, 0],
                [0.4 * cols, rows],
                [0.6 * cols, 0],
                [0.6 * cols, rows]], dtype=np.float32)
middle = transform.four_point_transform(full, pts)
middle_gray = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)

# apply a low pass filter and threahold it
middle_gray = cv2.GaussianBlur(middle_gray, (3, 15), 0)

# threshold the image and convert it to binary
middle_gray = cv2.adaptiveThreshold(middle_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# find the edges
middle_edge = cv2.Canny(middle_gray, 75, 200)
middle_edge = cv2.morphologyEx(middle_edge, cv2.MORPH_CLOSE, (25, 25))
cnt = middle.copy()

# (_, cnts,_) = cv2.findContours(middle_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
# for c in cnts:

#    peri = 0.02*cv2.arcLength(c, True)
#    approx = cv2.approxPolyDP(c,peri, True)
#    cv2.drawContours(cnt, [approx], -1, (0, 225, 0), 1)

# show image

minLineLength = 50
maxLineGap = 50
lines = cv2.HoughLinesP(middle_edge, 1, np.pi / 180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    continue

cv2.imshow('Image', cnt)
cv2.waitKey()
cv2.destroyAllWindows()

middle_row = (x1 + x2) / 2
middle_row = middle_row + 0.4 * cols
middle_row = middle_row * ratio
middle_row = int(middle_row)
orig_rows = int(rows * ratio)
orig_cols = int(cols * ratio)

# cv2.line(orig, (middle_row,0), (middle_row, orig_rows), (0,255,0), 2)
cv2.imshow("Original", orig)
cv2.waitKey()
cv2.destroyAllWindows()

pts1 = np.array([[0, 0],
                 [0, orig_rows],
                 [middle_row, 0],
                 [middle_row, orig_rows]], dtype=np.float32)

pts2 = np.array([[middle_row, 0],
                 [middle_row, orig_rows],
                 [orig_cols, 0],
                 [orig_cols, orig_rows]], dtype=np.float32)

img1 = transform.four_point_transform(orig, pts1)
img2 = transform.four_point_transform(orig, pts2)

cv2.imshow('First Page', img1)
cv2.imshow('Second Page', img2)
cv2.waitKey()
cv2.destroyAllWindows()
