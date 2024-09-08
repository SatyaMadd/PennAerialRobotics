import cv2
import numpy as np
import sys

def detectShapes(imagePath):
    shapeImage = cv2.imread(imagePath)
    # I black out the borders of the image it isn't determined as an edge.
    shapeImage[:, :5] = 0  
    shapeImage[:, -5:] = 0 
    shapeImage[:5, :] = 0  
    shapeImage[-5:, :] = 0  
  
    # I use Canny Edge detection here to detect edges in the image
    edges = cv2.Canny(shapeImage, 90, 40) 
    
    # I use morphological erosion and dilation to refine the edges detected in the image.
    # First, I apply dilation with a large kernel to close small gaps in the edges. 
    largeKernel = np.ones((10, 10), np.uint8)  
    dilatedEdges = cv2.dilate(edges, largeKernel, iterations=1)  

    # Then, I apply erosion with a smaller kernel to remove noise and refine the edge boundaries.
    smallKernel = np.ones((3, 3), np.uint8)  
    refinedEdges = cv2.erode(dilatedEdges, smallKernel, iterations=1) 
    
    # With the refined edges, I now find the Contours
    contours, hierarchy = cv2.findContours(refinedEdges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        approxShape = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        detailedShape = cv2.approxPolyDP(contour, 0.0005 * perimeter, True)
        
        # I check if the contour is a hole and has a significant area to determine if the contour is actually a shape
        isHole = hierarchy[0][i][3] != -1
        area = cv2.contourArea(contour)
        if area > 2500 and isHole and len(approxShape) < 18:
            cv2.drawContours(shapeImage, [detailedShape], 0, (200, 255, 0), 2)

            # Using image moments, I find the coordinates of the centroid of the contour
            moments = cv2.moments(contour)
            centerX = int(moments["m10"] / moments["m00"])
            centerY = int(moments["m01"] / moments["m00"])
            
            # I adjust coordinates relative to the image center, which is counted as (0,0) according to the sample solution
            height, width = shapeImage.shape[:2]
            adjustedCenterX = centerX - width // 2
            adjustedCenterY = centerY - height // 2
            coordinatesText = f"Coords: [{adjustedCenterX},{adjustedCenterY}]"
            cv2.putText(shapeImage, coordinatesText, (centerX - 80, centerY + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(shapeImage, (centerX, centerY), 3, (255, 255, 255), -1)
    
    # I resize the image for a better display
    height, width = shapeImage.shape[:2]
    newWidth = 800  
    newHeight = int((newWidth / width) * height)
    resizedImage = cv2.resize(shapeImage, (newWidth, newHeight))
    
    # I display the image with the shapes having outlines and with the coordinates of the centers shown
    cv2.imshow("Detected Shapes", resizedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if len(sys.argv) < 2:
    sys.exit(1)  
imagePath = 'ImageTests/' + sys.argv[1]
detectShapes(imagePath)