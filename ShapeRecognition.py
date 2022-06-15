from multiprocessing.reduction import duplicate
import cv2
from cv2 import threshold
import numpy as np

image = cv2.imread("Shapes.png") # Read the image as greyscale
image = cv2.resize(image, (1600,900)) # Resize the image to half the size
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to greyscale
gray = np.float32(image)
_, threshold = cv2.threshold(image, 100,255, cv2.THRESH_BINARY) # Threshold the image to pickup lighly colored images

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

                
                #cv2.drawContours(image, [c], 0, (0), 0) # Draw the contour of the shape

                mask = np.zeros(gray.shape, dtype="uint8") # Create a mask of the shape to preserve original image
                cv2.fillPoly(mask, [c], (255,255,255)) # Fill the shapes for easier corner detection
                dst = cv2.cornerHarris(mask,40,3,0.04) # Corner detection for each contour (40 block size for less precision)
                ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0) # Threshold the corner detection to get only the corners of the shape
                dst = np.uint8(dst)


                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst) # Connected components for corner detection
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001) # Criteria for corner detection
                corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria) # Store corner coordinates for each contour

                print(cv2.boundingRect(c), len(corners) - 1) # Print the bounding rectangle and the number of corners of the shape (arbitrarily large number for circle, +1 for any other shape.)
                if len(corners) == 4:
                    cv2.putText(image, "Triangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                elif len(corners) == 5:
                    if(abs(cv2.boundingRect(c)[2] - cv2.boundingRect(c)[3]) <= 15): # Comparison of width and height for square check
                        cv2.putText(image, "Square", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))
                    else:
                        cv2.putText(image, "Rectangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                elif len(corners) == 6:
                    cv2.putText(image, "Pentagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                elif len(corners) == 7:
                    cv2.putText(image, "Hexagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

                else:
                    cv2.putText(image, "Circle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()