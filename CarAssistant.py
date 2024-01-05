from __future__ import print_function
import cv2
import AppKit
#from google.cloud import vision
import time
import cloudmersive_image_api_client
from cloudmersive_image_api_client.rest import ApiException
from pprint import pprint
from ultralytics import YOLO

# Authenticate with Google Cloud Vision API
#client = vision.ImageAnnotatorClient()

model = YOLO("yolov8s.pt")
#model.train(data = "/Users/dorian/Downloads/road signs/data.yaml", epochs = 30)
modelTL = YOLO("best.pt")
# Open the video input
video = cv2.VideoCapture('/Users/dorian/Downloads/How To Handle Traffic Lights  |  Learn to drive_ Highway Code.mp4')

cnt = 0

redLight = False
rlFrames = 0
glFrames = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    cnt += 1
    # Convert the frame to an image
    #image = vision.Image(content=cv2.imencode('.jpg', frame)[1].tobytes())

    #transform frame to image
    #image = cv2.imencode('.jpg', frame)[1].tobytes()
    if cnt % 5 == 0:
        
        results = model.predict(frame)  # reduce size=320 for faster inference
        result = results[0]

        resultsTL = modelTL.predict(frame)  # reduce size=320 for faster inference
        resultTL = resultsTL[0]
        
        for box in result.boxes:
            
            cords = box.xyxy[0].tolist()
            cords = [int(x) for x in cords]
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            item = result.names[class_id]
            if item != "traffic light": 
                print("Object type:", item)
                print("Coordinates:", cords)
                print("Probability:", conf)
                cv2.rectangle(frame, tuple([cords[0],cords[1]]), tuple([cords[2], cords[3]]), (0, 255, 0), 2)
                cv2.putText(frame, result.names[class_id], (cords[0],cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        for box in resultTL.boxes:
           
            cords = box.xyxy[0].tolist()
            cords = [int(x) for x in cords]
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            item = resultTL.names[class_id]

            print("Object type:", item)
            print("Coordinates:", cords)
            print("Probability:", conf)
            cv2.rectangle(frame, tuple([cords[0],cords[1]]), tuple([cords[2], cords[3]]), (0, 255, 0), 2)
            cv2.putText(frame, resultTL.names[class_id], (cords[0],cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            if item == "red_light":
                rlFrames += 1
                if rlFrames > 15:
                    redLight = True
            if item == "green_light":
                glFrames += 1
                if redLight:
                    redLight = False
                    rlFrames = 0
                    glFrames = 0
                    print("Green light!")
                    cv2.putText(frame, "Goooooooooooooooooooooooooooooo", (cords[0],cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if redLight:
                print("Red light!")
            
               

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
video.release()
cv2.destroyAllWindows()

