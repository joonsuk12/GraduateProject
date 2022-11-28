from multiprocessing import RLock
import sys
import cv2
from cv2 import circle, waitKey
import numpy as np


# from flask import Flask, request, Response, jsonify
# from flask import render_template
# import base64
# from flask_cors import CORS
# from PIL import Image
# from io import BytesIO


confThreshold = 0.5
nmsThreshold = 0.4

# 테스트용도로 넣었던 코드
# app = Flask(__name__)
# CORS(app)
# @app.route('/dnn/yolo', methods=['GET', 'POST'])


def main(userVideoPath):
    # openpose 모델 & 설정 파일
    model1 = '/Users/LG/web-dnn/flask_barbell/model/pose_iter_440000.caffemodel'
    config1 = '/Users/LG/web-dnn/flask_barbell/model/pose_deploy_linevec.prototxt'

    # 포즈 점 개수, 점 연결 개수, 연결 점 번호 쌍
    nparts = 18
    npairs = 17
    pose_pairs = [(1, 2), (2, 3), (3, 4),  # 왼팔
                  (1, 5), (5, 6), (6, 7),  # 오른팔
                  (1, 8), (8, 9), (9, 10),  # 왼쪽다리
                  (1, 11), (11, 12), (12, 13),  # 오른쪽다리
                  (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # 얼굴

    #yolo모델 &설정파일
    model = '/Users/LG/web-dnn/flask_barbell/model/yolov3_custom_last.weights'
    config = '/Users/LG/web-dnn/flask_barbell/model/yolov3_bar.cfg'
    class_labels = '/Users/LG/web-dnn/flask_barbell/model/classes.names'

    # yolo 네트워크 설정
    net = cv2.dnn.readNet(model, config)
    print(3)
    if net.empty():
        print('Net open failed!')
        sys.exit()

    # openpose 네트워크 생성
    net2 = cv2.dnn.readNet(model1, config1)

    if net2.empty():
        print('Net open failed!')
        sys.exit()

    # 클래스 이름 불러오기

    classes = []
    with open(class_labels, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 출력 레이어 이름 받아오기

    layer_names = net.getLayerNames()  # 네트워크의 모든 레이어 이름을 가져와 layer_names에 넣음
    output_layers = [layer_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]
    # output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    # 동영상 열기
    cap = cv2.VideoCapture(userVideoPath)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)  # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))

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
    trackerK = cv2.TrackerCSRT_create()
    trackerF = cv2.TrackerCSRT_create()
    # 첫번째 프레임에서 추적 ROI 설정
    ret, frame = cap.read()

    cv2.imwrite('/Users/LG/web-dnn/flask_barbell/photo.jpg', frame)
    if not ret:
        print('Frame read failed!')
        sys.exit()
    img = cv2.imread('/Users/LG/web-dnn/flask_barbell/photo.jpg')

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
        #
        #
        #

    # 블롭 생성 & 추론
    blob2 = cv2.dnn.blobFromImage(img, 1/255., (368, 368))
    net2.setInput(blob2)
    out = net2.forward()  # out.shape = (1,57,46,46)

    h, w = img.shape[:2]

    # 검출된 점 추출
    points = []
    for i in range(nparts):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int(w * point[0] / out.shape[3])
        y = int(h * point[1] / out.shape[2])

        points.append((x, y) if conf > 0.1 else None)  # heat map threshold=0.1

    # 검출 결과 영상 만들기
    for pair in pose_pairs:
        p1 = points[pair[0]]
        p2 = points[pair[1]]

        if p1 is None or p2 is None:
            continue

        cv2.line(img, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 4, (0, 0, 255), -1, cv2.LINE_AA)
        # 오른다리 검출안되면 왼다리에서 좌표 검출하기
        if (pair[0] == 9 and pair[1] == 10):
            lab = "KneeR" + str(p1)
            ks = p1[0]
            ky = p1[1]
            fs = p2[0]
            fy = p2[1]
            cv2.putText(img, lab, (p1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)
        if (pair[0] == 12 and pair[1] == 13):
            lab = "KneeL" + str(p1)
            # if ks == 0 and ky == 0 : 여기서 좌표받기
            cv2.putText(img, lab, (p1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # 추론 시간 출력
        t, _ = net2.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite('/Users/LG/web-dnn/flask_barbell/photo.jpg', img)

    encodedVideoPath = '/Users/LG/web-dnn/flask_barbell/static/video/feedback.webm'
    # codec = 'theo'
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    encodedVideo = cv2.VideoWriter(encodedVideoPath, fourcc, fps, size)

    # 트래커시작
    rc = (sx, sy, bw, bh)
    rck = (ks, ky, bw, bh)
    rcf = (fs, fy, bw, bh)
    tracker.init(frame, rc)
    trackerK.init(frame, rck)
    trackerF.init(frame, rcf)

    circlelist = []
    before = 0

    # 매 프레임 처리
    while True:
        ret, frame = cap.read()

        if not ret:
            print('Frame read failed!')
            sys.exit()

        # 추적 & ROI 사각형 업데이트
        ret, rc = tracker.update(frame)
        ret, rck = trackerK.update(frame)
        ret, rcf = trackerF.update(frame)
        # 트래킹 동선기록을 저장하기 위한 튜플
        rcx = int(rc[0] + rc[2] / 2)
        rcy = int(rc[1] + rc[3] / 2)
        rc2 = (rcx, rcy)
        colorc = (0, before, 255)
        before = before + 1
        gett = rc2, colorc
        circlelist.append(gett)
        for i in circlelist:
            cv2.circle(frame, i[0], 4, i[1], -1, cv2.LINE_AA)

        rc = tuple([int(_) for _ in rc])
        cv2.rectangle(frame, rc, (0, 0, 255), 2)
        lab = "barbell_side" + \
            "(" + str(int(rc[0])) + "," + str(int(rc[1])) + ")"
        cv2.putText(frame, lab, (int(rc[0]), int(
            rc[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # 무릎과 발목의 위치좌표를 시각화하여 제공
        cv2.circle(frame, (int(rck[0]), int(rck[1])),
                   4, (0, 0, 255), -1, cv2.LINE_AA)
        labK = "KneeR"+"(" + str(int(rck[0])) + "," + str(int(rck[1])) + ")"
        cv2.putText(frame, labK, (int(rck[0]), int(
            rck[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (int(rcf[0]), int(rcf[1])),
                   4, (0, 0, 255), -1, cv2.LINE_AA)
        labF = "BASE" + "(" + str(int(rcf[0])) + "," + str(int(rcf[1])) + ")"
        cv2.putText(frame, labF, (int(rcf[0]), int(
            rcf[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # 바벨, 무릎, 발목의 각 좌표를 삼각형으로 그려 시각화
        if abs(int(rcf[0]) - int(rc[0])) > 40:
            cv2.line(frame, (int(rc[0]), int(rc[1])),
                     (int(rcf[0]), int(rcf[1])), (0, 0, 128))
            cv2.line(frame, (int(rc[0]), int(rc[1])),
                     (int(rck[0]), int(rck[1])), (0, 0, 128))
            cv2.line(frame, (int(rck[0]), int(rck[1])),
                     (int(rcf[0]), int(rcf[1])), (0, 0, 128))
        else:
            cv2.line(frame, (int(rc[0]), int(rc[1])),
                     (int(rcf[0]), int(rcf[1])), (127, 255, 0))
            cv2.line(frame, (int(rc[0]), int(rc[1])),
                     (int(rck[0]), int(rck[1])), (127, 255, 0))
            cv2.line(frame, (int(rck[0]), int(rck[1])),
                     (int(rcf[0]), int(rcf[1])), (127, 255, 0))
        #
        #
        label = str(rcf[0]-rc[0])
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        encodedVideo.write(frame)

        # if waitKey()==27:
        #   break
        if cv2.waitKey(25) == 45:
            return encodedVideoPath
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # app.run(debug=True, host='0.0.0.0')
