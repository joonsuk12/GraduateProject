import sys
from cv2 import circle, waitKey
import numpy as np
import cv2

# 모델 & 설정 파일
model = '/Users/junsuk/Desktop/python/LetMeSee/yolo_v3/yolov3_custom_last.weights'
config = '/Users/junsuk/Desktop/python/LetMeSee/yolo_v3/yolov3_bar.cfg'
class_labels = '/Users/junsuk/Desktop/python/LetMeSee/yolo_v3/classes.names'
confThreshold = 0.5
nmsThreshold = 0.4


# 네트워크 생성
net = cv2.dnn.readNet(model, config)
print("Program is running...")
if net.empty():
    print('Net open failed!')
    sys.exit()


# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 출력 레이어 이름 받아오기

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# 동영상 열기
cap = cv2.VideoCapture('/Users/junsuk/Desktop/python/LetMeSee/final.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)  # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로
if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 트래커 객체 생성

# Kernelized Correlation Filters
#tracker = cv2.TrackerKCF_create()

# Minimum Output Sum of Squared Error
#tracker = cv2.TrackerMOSSE_create()

# Discriminative Correlation Filter with Channel and Spatial Reliability
tracker = cv2.TrackerCSRT_create()

# 첫 번째 프레임에서 추적 ROI 설정
ret, frame = cap.read()

cv2.imwrite('/Users/junsuk/Desktop/python/LetMeSee/photo.jpg', frame)
if not ret:
    print('Frame read failed!')
    sys.exit()
img = cv2.imread('/Users/junsuk/Desktop/python/LetMeSee/photo.jpg')

if img is None:
    sys.exit()

# 블롭 생성 & 추론
blob = cv2.dnn.blobFromImage(img, 1/255., (320, 320), swapRB=True)
net.setInput(blob)
outs = net.forward(output_layers)

h, w = img.shape[:2]

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence)
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confThreshold:
            # 바운딩 박스 중심 좌표 & 박스 크기
            cx = int(detection[0] * w)
            cy = int(detection[1] * h)
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)

            # 바운딩 박스 좌상단 좌표
            sx = int(cx - bw / 2)
            sy = int(cy - bh / 2)

            boxes.append([sx, sy, bw, bh])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

# 비최대 억제
indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

for i in indices:
    i = i[0]
    sx, sy, bw, bh = boxes[i]
    label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
    color = colors[class_ids[i]]
    #cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
    #cv2.putText(img, label, (sx, sy - 10),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)







#트래커시작
rc = (sx,sy,bw,bh)

tracker.init(frame, rc)

circlelist=[]
before=0
# 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        print('Frame read failed!')
        sys.exit()

    # 추적 & ROI 사각형 업데이트
    ret, rc = tracker.update(frame)

    rcx = int(rc[0] + rc[2] / 2)
    rcy = int(rc[1] + rc[3] / 2)
    rc2 = (rcx, rcy)
    rc = tuple([int(_) for _ in rc])
    colorc=(0, before, 255)
    before=before+1
    gett=rc2,colorc
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    circlelist.append(gett)
    for i in circlelist:
        cv2.circle(frame, i[0], 2, i[1], -1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == 27:
        break
    
cv2.destroyAllWindows()
