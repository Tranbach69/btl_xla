import myLib.cannyLib_cus as canny_cus
import myLib.Roi_Draw as rd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('/home/cheater/opencv/image/MicrosoftTeams-image.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBinary=cv.threshold(gray_image,150,255,cv.THRESH_BINARY)[1]

# Printing image info
print(img.shape)
height = img.shape[0]
width = img.shape[1]

# Defining the required region
require_vertices = [
    (0, height),(0,height/2),
    (width, height/2),
    (width, height)
]

resultGauss = canny_cus.gaussian_smoothing(imgBinary, 7)
resultGradX = canny_cus.gradient_operation(resultGauss, 'x', 3, 7)
resultGradY = canny_cus.gradient_operation(resultGauss, 'y', 3, 7)
resultGradMag, resultGradAng = canny_cus.gradient_mag_and_angle(resultGradX, resultGradY, 3, 7)
resultNMS = canny_cus.non_maxima_suppression(resultGradMag, resultGradAng, 3, 7)
custom_canny=canny_cus.thresholding(resultNMS, resultGradMag, resultGradAng, 25, 3, 7)
# cv.imwrite('/home/cheater/opencv/image/custom_canny.jpg',custom_canny)

mask_image = rd.require_part(custom_canny, np.array([require_vertices], dtype=np.int32))

# Draw Lines
lines = cv.HoughLinesP(mask_image, rho=6, theta=np.pi / 70, threshold=160, lines=np.array([]), minLineLength=10, maxLineGap=5)

image_with_line = rd.draw_lines(img, lines)
# Showing the image using matplotlib
plt.imshow(image_with_line)
plt.show()
# cv.imshow("",custom_canny)
# cv.waitKey(0)