from ultralytics import YOLO
import cv2
import numpy as np

video = 'videos/short.webm'
weight_ball = 'models/s_1280_epoch1000.pt'
weight_player = 'models/yolov8x.pt'

model_ball = YOLO(weight_ball) 
model_player = YOLO(weight_player)

results_ball = model_ball.predict(video, stream=True, show=False, verbose=True, device=0, iou=0.1)  
results_players = model_player.predict(video, stream=True, show=False, verbose=True, device=0, classes=0, iou=0.45)  

total_frames = 0
detected_frames_balls = 0
detected_frames_players = 0

# Define HSV color ranges
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 40])
lower_yellow = np.array([23, 0, 0])
upper_yellow = np.array([29, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

for result_ball, result_player in zip(results_ball, results_players):
    total_frames += 1
    
    # Check for detected soccer balls and players
    if 'ball' in result_ball.names:
        detected_frames_balls += len(np.where(result_ball.boxes.cls == result_ball.names.index('ball'))[0])
    
    if 'person' in result_player.names:
        detected_frames_players += len(np.where(result_player.boxes.cls == result_player.names.index('person'))[0])

    # Combine and display results from both models
    img = result_ball.orig_img.copy()  # Extract the original image

    # Draw boxes for soccer balls
    for box in result_ball.boxes.xyxy:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 2)  # Red color for balls

    # Draw boxes for players
    for box in result_player.boxes.xyxy:
        x1, y1, x2, y2 = int(box[0]) + 10, int(box[1])- 10, int(box[2]), int(box[3])
        roi = img[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
        mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)

        if np.sum(mask_yellow) > np.sum(mask_white) and np.sum(mask_yellow) > np.sum(mask_red):
            label = "Referee"
            color = (0, 0, 0)  # Black color for referee
        elif np.sum(mask_white) > np.sum(mask_red):
            label = "Spain"
            color = (255, 255, 255)  # White color for Spain
        else:
            label = "Portugal"
            color = (0, 0, 255)  # Red color for Portugal

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    # Display the image
    cv2.imshow('Combined Detections', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)  # Display each frame for a short duration

cv2.destroyAllWindows()