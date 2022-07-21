import numpy as np
import matplotlib.pyplot as plt
from math import floor, degrees, atan2
import cv2 as cv

# Function for performing Convolution
def convolution(m1, m2, maskSum):
    n, m = m1.shape
    #print(m1,m2,maskSum)
    mSum = 0
    for i in range(n):
        for j in range(m):
            mSum += (int(m1[i, j]) * int(m2[i, j]))
    return int(round(mSum/maskSum))

# Function for getting gradient magnitude and angle from X & Y gradient
def gradient_mag_and_angle(gradX, gradY, gradientSize, gaussianSize):
    # Setting mask parameters
    outMag = np.zeros(gradX.shape, dtype='uint8')
    outAng = np.zeros(gradX.shape, dtype='float')
    # Variable to hold undefined area from gaussian filtering
    undefPixels = floor(gaussianSize/2) + floor(gradientSize/2)
    # Getting img dimensions
    n, m = gradX.shape
    # Starting gaussian masking
    for i in range(undefPixels, n-undefPixels):
        for j in range(undefPixels, m-undefPixels):
            outMag[i][j] = int(round((gradX[i][j]**2 + gradY[i][j]**2)**0.5))
            ang = degrees(atan2(gradY[i][j], gradX[i][j]))
            if ang < 0:    
                outAng[i][j] = ang + 360
            else:
                outAng[i][j] = ang
    return outMag, outAng

# Function for Gaussian Smoothing
def gaussian_smoothing(img, maskSize):
    # Setting mask parameters
    out = np.zeros(img.shape, dtype='uint8')
    if maskSize == 7:
        mask = np.array([[1,1,2,2,2,1,1],
                           [1,2,2,4,2,2,1],
                           [2,2,4,8,4,2,2],
                           [2,4,8,16,8,4,2],
                           [2,2,4,8,4,2,2],
                           [1,2,2,4,2,2,1],
                           [1,1,2,2,2,1,1]], dtype='uint8')
        maskSum = np.uint8(140)
        maskSizeHalf = floor(maskSize/2)
    # Getting img dimensions
    n, m = img.shape
    # Starting gaussian masking
    for i in range(n):
        for j in range(m):
            #i,j=3,223
            iLow = i - maskSizeHalf
            iUp = i + maskSizeHalf + 1
            jLow = j - maskSizeHalf
            jUp = j + maskSizeHalf + 1
            windowPixels = img[iLow:iUp, jLow:jUp]
            # 
            if windowPixels.shape == (maskSize, maskSize):
                #print(i,j)
                out[i,j] = convolution(mask, windowPixels, maskSum)
    return out
            
# Function for Gradient Operation
def gradient_operation(img, axis, maskSize, gaussianSize):
    # Setting mask parameters
    out = np.zeros(img.shape, dtype='int8')
    # Variable to hold undefined area from gaussian filtering
    undefPixels = floor(gaussianSize/2)
    # X axis sobel mask
    if maskSize == 3 and axis.lower() == 'x':
        mask = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]], dtype='int8')
        maskSizeHalf = floor(maskSize/2)
    # Y axis sobel mask
    if maskSize == 3 and axis.lower() == 'y':
        mask = np.array([[1,2,1],
                           [0,0,0],
                           [-1,-2,-1]], dtype='int8')
        maskSizeHalf = floor(maskSize/2)
    # Getting img dimensions
    n, m = img.shape
    # Starting gaussian masking
    for i in range(undefPixels+1, n-undefPixels-1):
        for j in range(undefPixels+1, m-undefPixels-1):
            #i,j,maskSizeHalf=4,4,1
            iLow = i - maskSizeHalf
            iUp = i + maskSizeHalf + 1
            jLow = j - maskSizeHalf
            jUp = j + maskSizeHalf + 1
            windowPixels = img[iLow:iUp, jLow:jUp]
            # 
            if windowPixels.shape == (maskSize, maskSize):
                #print(i,j)
                out[i,j] = convolution(mask, windowPixels, 1)
    return out

def non_maxima_suppression(gradMag, gradAng, gradientSize, gaussianSize):
    # Setting mask parameters
    out = np.zeros(gradMag.shape, dtype='uint8')
    # Variable to hold undefined area from gaussian filtering
    # To process 8-connected neighbors of NMS,1 more pixel is added to border
    undefPixels = floor(gaussianSize/2) + floor(gradientSize/2) + 1
    # Getting img dimensions
    n, m = gradMag.shape
    # Starting gaussian masking
    for i in range(undefPixels, n-undefPixels):
        for j in range(undefPixels, m-undefPixels):
            # Sector 0 neighbors
            if (337.5<=gradAng[i][j]<=360) or (0<=gradAng[i][j]<22.5) or (157.5<=gradAng[i][j]<202.5):
                neighbor1 = gradMag[i][j+1]
                neighbor2 = gradMag[i][j-1]
            # Sector 1 neighbors
            elif (22.5<=gradAng[i][j]<67.5) or (202.5<=gradAng[i][j]<247.5):
                neighbor1 = gradMag[i-1][j+1]
                neighbor2 = gradMag[i+1][j-1]
            # Sector 2 neighbors
            elif (67.5<=gradAng[i][j]<112.5) or (247.5<=gradAng[i][j]<292.5):
                neighbor1 = gradMag[i-1][j]
                neighbor2 = gradMag[i+1][j]
            # Sector 3 neighbors
            elif (112.5<=gradAng[i][j]<157.5) or (292.5<=gradAng[i][j]<337.5):
                neighbor1 = gradMag[i-1][j-1]
                neighbor2 = gradMag[i+1][j+1]
            # Default case
            else:
                neighbor1 = 255
                neighbor2 = 255
            # Comparing current pixel with its neighbors
            if gradMag[i][j] > neighbor1 and gradMag[i][j] > neighbor2:
                out[i][j] = gradMag[i][j]
    return out
    

def thresholding(nmsImg, gradMag, gradAng, t, gradientSize, gaussianSize):
    # Setting mask parameters
    out = np.zeros(nmsImg.shape, dtype='uint8')
    # Setting threshold values
    t1 = t
    t2 = 2*t
    # Variable to hold undefined area from gaussian filtering
    # To process 8-connected neighbors of NMS,1 more pixel is added to border
    undefPixels = floor(gaussianSize/2) + floor(gradientSize/2) + 1
    # Getting img dimensions
    n, m = nmsImg.shape
    # Starting gaussian masking
    for i in range(undefPixels, n-undefPixels):
        for j in range(undefPixels, m-undefPixels):
            
            # Case 1 with pixel less than T1
            if nmsImg[i][j] < t1:
                out[i][j] = 0
            # Case 2 with pixel between T1 and T2
            elif t1 <= nmsImg[i][j] <= t2:
                # Extracting 8-neighbor magnitudes
                neighborMags = [gradMag[i-1,j-1], gradMag[i-1,j], gradMag[i-1,j+1],
                                gradMag[i,j-1], gradMag[i,j+1],
                                gradMag[i+1,j-1], gradMag[i+1,j], gradMag[i+1,j+1]]
                # Extracting 8-neighbor angles
                neighborAngs = [gradAng[i-1,j-1], gradAng[i-1,j], gradAng[i-1,j+1],
                                gradAng[i,j-1], gradAng[i,j+1],
                                gradAng[i+1,j-1], gradAng[i+1,j], gradAng[i+1,j+1]]
                # Comparing the 8-neighbors of the current pixel with T2
                val = 0
                for k in range(8):
                    #print(neighborMags[k], abs(neighborAngs[k] - gradAng[i][j]))
                    if neighborMags[k] > t2 and abs(neighborAngs[k] - gradAng[i][j]) <= 45:
                        val = 255
                        break
                out[i][j] = val
            # Case 3 with pixel greater than T2
            elif nmsImg[i][j] > t2:
                out[i][j] = 255
    return out
# MAIN

# img = cv.imread("/home/cheater/opencv/image/rosé.jpg", 0) 
# resultGauss = gaussian_smoothing(img, 7)
# resultGradX = gradient_operation(resultGauss, 'x', 3, 7)
# resultGradY = gradient_operation(resultGauss, 'y', 3, 7)
# resultGradMag, resultGradAng = gradient_mag_and_angle(resultGradX, resultGradY, 3, 7)
# resultNMS = non_maxima_suppression(resultGradMag, resultGradAng, 3, 7)
# custom_canny = thresholding(resultNMS, resultGradMag, resultGradAng, 25, 3, 7)

#imwrite('custom_canny.jpg', custom_canny)
