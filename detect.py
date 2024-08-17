from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("Video/PPEL.mp4") 
model = YOLO("runs/detect/train10/weights/best.pt")
#model = YOLO("PPE.pt")
classNames = ['Gloves', 'Helmet', 'No Gloves', 'No Helmet', 'No Vest', 'Vest', 'person', 'shoes']
#classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)
target_width, target_height = 852, 480  # 1080p resolution

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize the frame
    img = cv2.resize(img, (target_width, target_height))

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and Class Name (same as before)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                # Determine color based on the detected class
                if currentClass in ['No Gloves', 'No Helmet', 'No Vest']:
                #if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    myColor = (0, 0, 255) # Red
                elif currentClass in ['Gloves', 'Helmet', 'Vest', 'shoes']:
                #elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    myColor = (0, 255, 0) # Green
                else:
                    myColor = (255, 0, 0) # Blue

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                  (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                  colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
