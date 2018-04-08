import cv2
import numpy as np
import imutils
import  transform

#reading the image
img = cv2.imread('dualpage.jpg')

#reduce dimensions
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)


#convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#apply a low pass filter
gray = cv2.GaussianBlur(gray , (15,15), 0)

#threshold the image and convert it to binary
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 , 2)

#find the edges
edge = cv2.Canny(gray, 75, 200)

#fil the space
#edge1 = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, (25,25), iterations = 1)
edge = cv2.dilate(edge, (101,101))

#show image
cv2.imshow('edge image', edge)
cv2.waitKey()
cv2.destroyAllWindows()

#draw contours
(_, cnts,_) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
for c in cnts:
    peri = 0.02*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,peri, True)

    #the book has approx 4 sides
    if(len(approx) == 4):
        screenCnt = approx
        break;

#draw Contour
cnt = img.copy()
cv2.drawContours(cnt, [screenCnt], -1, (0,225,0) , 2)
cv2.imshow('Book',cnt )
cv2.waitKey()
cv2.destroyAllWindows()

#perspective trasform
# apply the four point transform to obtain a top-down
# view of the original image
origCnt = screenCnt.reshape(4, 2) * ratio
warped = transform.four_point_transform(orig,origCnt)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = cv2.adaptiveThreshold(warped,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey()

#write it into an image
cv2.imwrite('scanned.jpg', warped)