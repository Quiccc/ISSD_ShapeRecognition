from multiprocessing.reduction import duplicate
import cv2
from cv2 import threshold
import numpy as np

def imageFilter(imageName, contourThreshold):
    originalImage = cv2.imread("Shape.jpg")
    originalImage = cv2.resize(originalImage, (1600,900))
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    Intensity_Matrix = np.ones(originalImage.shape, dtype="uint8") * 90 #Scalar value for intensity of the image
    if(imageName == "Bright Image"):
        editedImage = cv2.add(originalImage, Intensity_Matrix) # Add the matrix to the image to make it brighter

    else:
        editedImage = cv2.subtract(originalImage, Intensity_Matrix) # Subtract the matrix from the image to make it darker

    editedImage = showShapes(editedImage, contourThreshold) # Call the function to show the shapes

    cv2.imshow(imageName, editedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showShapes(image, contourThreshold):
    gray = np.float32(image)
    _, threshold = cv2.threshold(image, contourThreshold,255, cv2.THRESH_BINARY) # Threshold for contour detection (210 for bright, 20 for dark)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find the contours of the image
    contours = [c for i, c in enumerate(contours) if i % 2 == 0] # Remove the similar contours

    for c in contours:

        M = cv2.moments(c) # Calculate the moments of the contour to calculate the center of a shape
        if M["m00"] != 0:  # If the shape area is not 0
            x = int(M["m10"] / M["m00"]) # x-coordinate of center of shape
            y = int(M["m01"] / M["m00"]) # y-coordinate of center of shape

            if cv2.boundingRect(c)[0] == 0: # To skip drawing contours for the edge of the entire image
                continue

            else:
                if cv2.boundingRect(c)[2] > 10 and cv2.boundingRect(c)[3] > 10: # To skip drawing contours for small shapes

                
                    cv2.drawContours(image, [c], 0, (0), 5) # Draw the contour of the shape

                    mask = np.zeros(gray.shape, dtype="uint8") # Create a mask of the shape to preserve original image
                    cv2.fillPoly(mask, [c], (255,255,255)) # Fill the shapes for easier corner detection
                    dst = cv2.cornerHarris(mask,40,3,0.04) # Corner detection for each contour (40 block size to get more accurate corner locations)
                    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0) # Threshold the corner detection to get only the corners of the shape
                    dst = np.uint8(dst)


                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst) # Connected components for corner detection
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001) # Criteria for corner detection
                    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria) # Store corner coordinates for each contour

                    print(cv2.boundingRect(c), len(corners) - 1) # Print the bounding rectangle and the number of corners of the shape (arbitrarily large number for circle, +1 for any other shape.)
                    if len(corners) == 4:
                        cv2.putText(image, "Triangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                    elif len(corners) == 5:
                        if(abs(cv2.boundingRect(c)[2] - cv2.boundingRect(c)[3]) <= 25): # Comparison of width and height for square check
                            cv2.putText(image, "Square", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))
                        else:
                            cv2.putText(image, "Rectangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                    elif len(corners) == 6:
                        cv2.putText(image, "Pentagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                    elif len(corners) == 7:
                        cv2.putText(image, "Hexagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                    else:
                        cv2.putText(image, "Circle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))
                    
    return image

imageFilter("Bright Image", 200)
imageFilter("Dark Image", 20)