import cv2
import numpy as np
import time
import sys

#Mostly same as ShapeDetection, differences commented
def videoShapeDetector(videoPath):
    videoFile = cv2.VideoCapture(videoPath)
    prevFrame = 0
    newFrame = 0
    
    while videoFile.isOpened():
        ret, frame = videoFile.read()
        if not ret:
            break

        resultFrame = frame.copy()
        
        frame[:, :5] = 0  
        frame[:, -5:] = 0 
        frame[:5, :] = 0  
        frame[-5:, :] = 0  
        
        # Use Canny Edge detection to detect edges in the frame
        edges = cv2.Canny(frame, 90, 40)  
        

        # Used closing operation instead of dilation and erosion, worked better in video but doesn't identify the shape in the first sample image.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  
        refinedEdges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)   
        
        contours, hierarchy = cv2.findContours(refinedEdges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            approximateShape = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            detShape = cv2.approxPolyDP(contour, 0.0005 * perimeter, True)
            isHole = hierarchy[0][i][3] != -1            
            area = cv2.contourArea(contour)
            if area > 7000 and isHole and len(approximateShape) < 20:
                cv2.drawContours(resultFrame, [detShape], 0, (200, 255, 0), 2)                
                moments = cv2.moments(contour)
                coordX = int(moments["m10"] / moments["m00"])
                coordY = int(moments["m01"] / moments["m00"])                
                height, width = frame.shape[:2]
                adjustedCoordX = coordX - width // 2
                adjustedCoordY = coordY - height // 2
                coordText = f"Coords: [{adjustedCoordX},{adjustedCoordY}]"                
                cv2.putText(resultFrame, coordText, (coordX - 180, coordY + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.circle(resultFrame, (coordX, coordY), 3, (255, 255, 255), -1)

        # Since speed of application was something that was being evaluated, I added FPS to video
        newFrame = time.time()
        fps = 1 / (newFrame - prevFrame)
        prevFrame = newFrame
        fps = int(fps)
        fpsText = "FPS=" + str(fps)
        cv2.putText(resultFrame, fpsText, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 0), 3) 

        height, width = resultFrame.shape[:2]
        newWidth = 800  
        newHeight = int((newWidth / width) * height)
        resized_frame = cv2.resize(resultFrame, (newWidth, newHeight))
        
        # I display each frame with detected shapes and FPS as I encounter each frame in the video, with video being closed with q
        cv2.imshow("Video", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    videoFile.release()
    cv2.destroyAllWindows()

if len(sys.argv) < 2:
    sys.exit(1)  
videoPath = 'VideoTests/' + sys.argv[1]
videoShapeDetector(videoPath)