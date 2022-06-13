import cv2
from cv2 import threshold
import numpy as np

image = cv2.imread("Shapes.jpg", cv2.IMREAD_GRAYSCALE) # Read the image as greyscale
_, threshold = cv2.threshold(image, 220,255, cv2.THRESH_BINARY) # Threshold the image to pickup lighly colored images
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find the contours of the image

for c in contours:

    M = cv2.moments(c) # Calculate the moments of the contour to calculate the center of a shape
    if M["m00"] != 0:  # If the shape area is not 0
        x = int(M["m10"] / M["m00"]) # x-coordinate of center of shape
        y = int(M["m01"] / M["m00"]) # y-coordinate of center of shape

        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True) # Smooth the contour to make it more accurate
        print(cv2.boundingRect(approx))
        if cv2.boundingRect(approx)[0] == 0: # To skip drawing contours for the edge of the entire image
            continue

        else:
            cv2.drawContours(image, [approx], 0, (0), 5) # Draw the contour of the shape
            if len(approx) == 3:
                cv2.putText(image, "Triangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

            elif len(approx) == 4:
                if(abs(cv2.boundingRect(approx)[2] - cv2.boundingRect(approx)[3]) <= 2): # Comparison of width and height for square check
                    cv2.putText(image, "Square", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))
                else:
                    cv2.putText(image, "Rectangle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

            elif len(approx) == 5:
                cv2.putText(image, "Pentagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

            elif len(approx) == 6:
                cv2.putText(image, "Hexagon", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))

            else:
                cv2.putText(image, "Circle", (x - 20, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()