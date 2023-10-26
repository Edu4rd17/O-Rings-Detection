# Image Processing Project - B00125295 - Eduard Iacob
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import queue


def threshold(img, thresh):
    # loop through the rows
    for y in range(0, img.shape[0]):
        # loop through the columns
        for x in range(0, img.shape[1]):
            # check if the current pixel is greater than the threshold, if it is set the current pixel to 0, else set the current pixel to 255
            if img[y, x] > thresh:
                img[y, x] = 0
            else:
                img[y, x] = 255
    return img


def binaryMorphologyDilation(img):
    # create a copy of the image
    copyImage = img.copy()
    # loop through the rows
    for y in range(1, img.shape[0]-1):
        # loop through the columns
        for x in range(1, img.shape[1]-1):
            # check the surrounding neighbors of the pixel to see if they are 0, if they are set the current pixel to 0
            if img[y, x] == 0:
                if img[y-1, x-1] == 255 or img[y-1, x] == 255 or img[y-1, x+1] == 255 or img[y, x-1] == 255 or img[y, x] == 255 \
                        or img[y, x+1] == 255 or img[y+1, x-1] == 255 or img[y+1, x] == 255 or img[y+1, x+1] == 255:
                    copyImage[y, x] = 255
    return copyImage


def binaryMorphologyErosion(img):
    # create a copy of the image
    copyImage = img.copy()
    # loop through the rows
    for y in range(1, img.shape[0]-1):
        # loop through the columns
        for x in range(1, img.shape[1]-1):
            # check the surrounding neighbors of the pixel to see if they are 255, if they are set the current pixel to 255,
            # basically erosion is the opposite of dilation
            if img[y, x] == 255:
                if img[y-1, x-1] == 0 or img[y-1, x] == 0 or img[y-1, x+1] == 0 or img[y, x-1] == 0 or img[y, x] == 0 or img[y, x+1] == 0 \
                        or img[y+1, x-1] == 0 or img[y+1, x] == 0 or img[y+1, x+1] == 0:
                    copyImage[y, x] = 0
    return copyImage


def componentLabelingForeground(img):
    # create an array to store the labels
    labelArray = np.zeros(img.shape, dtype=np.uint8)
    # define the current label
    currentLabel = 1
    # create a queue to store the pixels
    queuee = queue.Queue()
    # loop through the rows
    for y in range(1, img.shape[0]-1):
        # loop through the columns
        for x in range(1, img.shape[1]-1):
            # check if the current pixel is 0 and the copy image is 0, if it is set the current pixel to the current label
            if img[y, x] == 255 and labelArray[y, x] == 0:
                labelArray[y, x] = currentLabel
                # add the current pixel to the queue
                queuee.put((y, x))
                # loop through the queue
                while not queuee.empty():
                    # assign the x and y coordinates of the pixel to variables
                    queue_y, queue_x = queuee.get()
                    # check the surrounding neighbors of the pixel to see if they are 0 and the copy image is 0, if they are set the current pixel to the current label
                    if img[queue_y-1, queue_x-1] == 255 and labelArray[queue_y-1, queue_x-1] == 0:
                        labelArray[queue_y-1, queue_x-1] = currentLabel
                        queuee.put((queue_y-1, queue_x-1))
                    if img[queue_y-1, queue_x] == 255 and labelArray[queue_y-1, queue_x] == 0:
                        labelArray[queue_y-1, queue_x] = currentLabel
                        queuee.put((queue_y-1, queue_x))
                    if img[queue_y-1, queue_x+1] == 255 and labelArray[queue_y-1, queue_x+1] == 0:
                        labelArray[queue_y-1, queue_x+1] = currentLabel
                        queuee.put((queue_y-1, queue_x+1))
                    if img[queue_y, queue_x-1] == 255 and labelArray[queue_y, queue_x-1] == 0:
                        labelArray[queue_y, queue_x-1] = currentLabel
                        queuee.put((queue_y, queue_x-1))
                    if img[queue_y, queue_x+1] == 255 and labelArray[queue_y, queue_x+1] == 0:
                        labelArray[queue_y, queue_x+1] = currentLabel
                        queuee.put((queue_y, queue_x+1))
                    if img[queue_y+1, queue_x-1] == 255 and labelArray[queue_y+1, queue_x-1] == 0:
                        labelArray[queue_y+1, queue_x-1] = currentLabel
                        queuee.put((queue_y+1, queue_x-1))
                    if img[queue_y+1, queue_x] == 255 and labelArray[queue_y+1, queue_x] == 0:
                        labelArray[queue_y+1, queue_x] = currentLabel
                        queuee.put((queue_y+1, queue_x))
                    if img[queue_y+1, queue_x+1] == 255 and labelArray[queue_y+1, queue_x+1] == 0:
                        labelArray[queue_y+1, queue_x+1] = currentLabel
                        queuee.put((queue_y+1, queue_x+1))
                currentLabel += 1

    # return the label array and the current label
    return labelArray, (currentLabel - 1)


def componentLabelingBackground(img):
    # create an array to store the labels
    labelArray = np.zeros(img.shape, dtype=np.uint8)
    # define the current label
    currentLabel = 1
    # create a queue to store the pixels
    queuee = queue.Queue()
    # loop through the rows
    for y in range(1, img.shape[0]-1):
        # loop through the columns
        for x in range(1, img.shape[1]-1):
            # check if the current pixel is 0 and the copy image is 0, if it is set the current pixel to the current label
            if img[y, x] == 0 and labelArray[y, x] == 0:
                labelArray[y, x] = currentLabel
                # add the current pixel to the queue
                queuee.put((y, x))
                # loop through the queue
                while not queuee.empty():
                    # assign the x and y coordinates of the pixel to variables
                    queue_y, queue_x = queuee.get()
                    # check the surrounding neighbors of the pixel to see if they are 0 and the copy image is 0, if they are set the current pixel to the current label
                    if img[queue_y-1, queue_x-1] == 0 and labelArray[queue_y-1, queue_x-1] == 0:
                        labelArray[queue_y-1, queue_x-1] = currentLabel
                        queuee.put((queue_y-1, queue_x-1))
                    if img[queue_y-1, queue_x] == 0 and labelArray[queue_y-1, queue_x] == 0:
                        labelArray[queue_y-1, queue_x] = currentLabel
                        queuee.put((queue_y-1, queue_x))
                    if img[queue_y-1, queue_x+1] == 0 and labelArray[queue_y-1, queue_x+1] == 0:
                        labelArray[queue_y-1, queue_x+1] = currentLabel
                        queuee.put((queue_y-1, queue_x+1))
                    if img[queue_y, queue_x-1] == 0 and labelArray[queue_y, queue_x-1] == 0:
                        labelArray[queue_y, queue_x-1] = currentLabel
                        queuee.put((queue_y, queue_x-1))
                    if img[queue_y, queue_x+1] == 0 and labelArray[queue_y, queue_x+1] == 0:
                        labelArray[queue_y, queue_x+1] = currentLabel
                        queuee.put((queue_y, queue_x+1))
                    if img[queue_y+1, queue_x-1] == 0 and labelArray[queue_y+1, queue_x-1] == 0:
                        labelArray[queue_y+1, queue_x-1] = currentLabel
                        queuee.put((queue_y+1, queue_x-1))
                    if img[queue_y+1, queue_x] == 0 and labelArray[queue_y+1, queue_x] == 0:
                        labelArray[queue_y+1, queue_x] = currentLabel
                        queuee.put((queue_y+1, queue_x))
                    if img[queue_y+1, queue_x+1] == 0 and labelArray[queue_y+1, queue_x+1] == 0:
                        labelArray[queue_y+1, queue_x+1] = currentLabel
                        queuee.put((queue_y+1, queue_x+1))
                currentLabel += 1

    return labelArray


def detectFaultyOring(numLabels, img):
    # get the process time
    cv.putText(image, "Time: " + str(after-before)[:-3] + " sec", (5, 210),
               cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))
    # print(numLabels)
    # check if the num of labels is not 1 or the number of unique values is not 3, then the oring is faulty
    if numLabels != 1:
        cv.putText(image, "Fail: Chipped piece", (15, 16),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))

    elif len(np.unique(img)) != 3:
        cv.putText(image, "Fail: Broken o-ring", (15, 16),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
    # else the oring is not faulty
    else:
        cv.putText(image, "Pass: Not Faulty", (15, 16),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0))


def drawRadiusLineOfOring(numLabels, imageFore):
    # get the pixels that are on the outer edge of the oring
    edgePixels = np.zeros(imageFore.shape, dtype=np.uint8)
    # loop through the number of labels
    for i in range(1, numLabels+1):
        # get the x and y coordinates of the pixels that are equal to the current label
        y, x = np.where(imageFore == i)
        # check if the size of the x and y coordinates are not 0
        if x.size != 0 or y.size != 0:
            # loop through the x and y coordinates
            for j in range(len(x)):
                # check if the surrounding pixels are not equal to the current label
                if imageFore[y[j]-1, x[j]-1] != i or imageFore[y[j]-1, x[j]] != i or imageFore[y[j]-1, x[j]+1] != i or imageFore[y[j], x[j]-1] != i or imageFore[y[j], x[j]+1] != i or imageFore[y[j]+1, x[j]-1] != i or imageFore[y[j]+1, x[j]] != i or imageFore[y[j]+1, x[j]+1] != i:
                    # set the pixel to 255
                    edgePixels[y[j], x[j]] = 255
                    # draw a circle on the pixel
                    cv.circle(image, (x[j], y[j]), 0, (0, 255, 0), 0)


def getCenterOfOring(numLabels, imageFore):
    # loop through the number of labels
    for i in range(1, numLabels+1):
        # get the x and y coordinates of the pixels that are equal to the current label
        y, x = np.where(imageFore == i)
        # check if the size of the x and y coordinates are not 0
        if x.size != 0 or y.size != 0:
            # get the center of the pixels
            center = (int(np.mean(x)), int(np.mean(y)))
            # draw a circle on the center of the pixels
            cv.circle(image, center, 8, (255, 0, 0), 0)


def imhist(img):
    hist = np.zeros(256)
    for y in range(0, img.shape[0]):  # loops through the rows
        for x in range(0, img.shape[1]):  # loops through the columns
            hist[img[y, x]] += 1
    return hist


def findT(hist):
    max_value = 0
    max_index = -1
    for i in range(0, hist.shape[0]):
        if hist[i] > max_value:
            max_value = hist[i]
            max_index = i
    return max_index - 50


# read in the image into memory
i = 1
while True:
    image = cv.imread('Oring' + str(i) + '.jpg', 0)
    i += 1
    if i == 16:
        i = 1
    hist = imhist(image)
    # plt.plot(hist)
    before = time.time()
    thresh = findT(hist)
    bwImage = threshold(image.copy(), thresh)
    after = time.time()
    # cv.imshow('binary', bwImage)
    morphologyImage = binaryMorphologyDilation(bwImage)
    morphologyImage = binaryMorphologyErosion(morphologyImage)
    compLabelImageFore, numLabels = componentLabelingForeground(
        morphologyImage)
    compLabelImageBack = componentLabelingBackground(morphologyImage)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    detectFaultyOring(numLabels, compLabelImageBack)
    getCenterOfOring(numLabels, compLabelImageFore)
    drawRadiusLineOfOring(numLabels, compLabelImageFore)
    cv.imshow('Original', image)
    cv.imshow('Dilation', bwImage)
    cv.imshow('Erosion', morphologyImage)
    cv.imshow('Morphology', morphologyImage)
    # cv.imshow('Component Labeling', compLabelImageFore.astype(np.uint8))
    # label = plt.imshow(compLabelImageFore)
    plt.show()
    c = cv.waitKey(0)
    if c & 0XFF == ord('q'):
        break
