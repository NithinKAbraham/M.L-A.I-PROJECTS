import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time
import winsound  
import math

model = YOLO('yolov8s.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'gun', 'rifle', 'knife']

tracker = Tracker()
count = 0

cap = cv2.VideoCapture('People in Public Space _ Copyright Free Video Footage.mp4')

down = {}
up = {}

persons_in_green = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020,500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    
    objects_rect = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row[:6])
        c = class_list[d]
        if 'person' in c or 'gun' in c or 'rifle' in c or 'knife' in c or 'backpack' in c or 'handbag' in c or 'suitcase' in c:
            objects_rect.append((x1, y1, x2 - x1, y2 - y1))

    
    objects_bbs_ids = tracker.update(objects_rect)

    for obj_bb_id in objects_bbs_ids:
        x3, y3, w, h, id = obj_bb_id
        cx = (x3 + x3 + w) // 2
        cy = (y3 + y3 + h) // 2

        red_line_y = 70
        blue_line_y = 368
        green_line1_y = 80 
        green_line2_y = 355
        offset = 7

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        if 'gun' in class_list[id] or 'rifle' in class_list[id] or 'knife' in class_list[id] or 'backpack' in class_list[id] or 'handbag' in class_list[id] or 'suitcase' in class_list[id]:
            cv2.rectangle(frame, (x3, y3), (x3 + w, y3 + h), (148, 0, 211), 2) 
        else:
            cv2.rectangle(frame, (x3, y3), (x3 + w, y3 + h), (0, 0, 0), 2)
        
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        
        if green_line1_y < (cy + offset) < green_line2_y or green_line1_y < (cy - offset) < green_line2_y:
            if id not in down:
                persons_in_green += 1
                down[id] = True
        else:
            if id in down:
                persons_in_green -= 1
                del down[id]

    
        if green_line2_y < cy < blue_line_y and id in down:
            winsound.Beep(1000, 200) 

    text_color1 = (0, 0, 255)
    text_color2 = (255, 0, 0)
    text_color3 = (255, 225, 0)

    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)

    cv2.line(frame, (8, 70), (1300, 70), red_color, 1)
    cv2.putText(frame, ('ENTERY_LINE'), (685, 65), cv2.FONT_HERSHEY_SIMPLEX, .9, text_color1, 1, cv2.LINE_AA)

    cv2.line(frame, (8, 368), (1300, 368), blue_color, 1)
    cv2.putText(frame, ('RESTRICTED_AREA_LINE'), (685, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color2, 1, cv2.LINE_AA)

    cv2.line(frame, (8, green_line1_y), (1300, green_line1_y), green_color, 1)
    cv2.line(frame, (8, green_line2_y), (1300, green_line2_y), green_color, 1)

    cv2.putText(frame, f'Persons in Green Zone: {persons_in_green}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color3, 2)

    cv2.imshow("CCTV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()