from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("YOLO-segment-v3/train1/weights/best.pt")

# object classes
classNames = [
  'bottled_soda',        # ID 1
  'cheese',              # ID 2
  'chocolate',           # ID 3
  'coffee',              # ID 4
  'condensed_milk',      # ID 5
  'cooking_oil',         # ID 6
  'corned_beef',         # ID 7
  'garlic',              # ID 8
  'instant_noodles',     # ID 9
  'ketchup',             # ID 10
  'lemon',               # ID 11
  'boxed_cream',         # ID 12
  'mayonnaise',          # ID 13
  'peanut_butter',       # ID 14
  'pasta',               # ID 15
  'pineapple_juice',     # ID 16
  'crackers',            # ID 17
  'canned_sardines',     # ID 18
  'bottled_shampoo',     # ID 19
  'soap',                # ID 20
  'soy_sauce',           # ID 21
  'toothpaste',          # ID 22
  'canned_tuna',         # ID 23
  'isopropyl_alcohol'    # ID 24
]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()