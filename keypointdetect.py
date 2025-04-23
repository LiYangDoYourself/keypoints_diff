# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/9 16:53
import json

import cv2
import time
from ultralytics import YOLO


# keynamedcit={"nose":0,"lefteye":1,"leftrear":3,"leftshoulder":5,"leftelbow":7,"leftwrist":9,"lefthip":11,"leftknee":13,"leftankle":15,
#              "righteye":2,"rightrear":4,"rightshoulder":6,"rightelbow":8,"rightwrist":10,"righthip":12,"rightknee":14,"rightankle":16}


score_conf=0.5
pic_scale=1280
model_path = "r'./best.pt'"

keynamedcit={
    0: "nose",          # 鼻子
    1: "lefteye",       # 左眼
    2: "righteye",      # 右眼
    3: "leftrear",      # 左耳
    4: "rightrear",     # 右耳
    5: "leftshoulder",  # 左肩
    6: "rightshoulder", # 右肩
    7: "leftelbow",     # 左肘
    8: "rightelbow",    # 右肘
    9: "leftwrist",     # 左腕
    10: "rightwrist",   # 右腕
    11: "lefthip",      # 左髋
    12: "righthip",     # 右髋
    13: "leftknee",     # 左膝
    14: "rightknee",    # 右膝
    15: "leftankle",    # 左踝
    16: "rightankle"    # 右踝
}

def modelinit():
    model = YOLO(model_path)
    return model

def framedetect(frame,model):
    # 使用YOLOv8模型进行检测
    results = model(frame, conf=score_conf, imgsz=pic_scale)
    return results


def optimize_keypoints(frameid_keypoint, start_kpt=5, end_kpt=16):
    total_frames = len(frameid_keypoint)

    for k in range(total_frames):
        current_frame = frameid_keypoint[k]

        for i in range(start_kpt, end_kpt + 1):
            if i >= len(current_frame) - 1:
                continue  # 避免越界，最后一个是 box

            x, y = current_frame[i]

            if (x, y) == (0, 0):
                # 尝试从前一帧取
                if k > 0 and frameid_keypoint[k - 1][i] != (0, 0):
                    current_frame[i] = frameid_keypoint[k - 1][i]
                # 再尝试从后一帧取
                elif k < total_frames - 1 and frameid_keypoint[k + 1][i] != (0, 0):
                    current_frame[i] = frameid_keypoint[k + 1][i]

        frameid_keypoint[k] = current_frame

    return frameid_keypoint


if __name__ == '__main__':


    # first_frame = []
    frameid_keypoint = {}
    model = YOLO(r'./best.pt')

    modelnames = model.names
    print("modelnames:",modelnames)

    # cap = cv2.VideoCapture(r"rtsp://admin:798446835qqcom@192.168.1.64:554/h264/ch1/main/av_stream")
    cap = cv2.VideoCapture(r"H:\workspace\keypoints_diff\testvideos\result20250411094252-test.mp4")

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    k = 0
    while True:
        start_time = time.time()  # 开始时间

        ret, frame = cap.read()

        if not ret:
            print("无法接收帧")
            break
            # continue

        # 使用YOLOv8模型进行检测
        results = model(frame,conf=0.5,imgsz=1280)


        # 提取检测结果并绘制在原图上
        # for result in results:
        #     for box in result.boxes:
        #         # 提取边界框坐标
        #         x1, y1, x2, y2 = box.xyxy[0]
        #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        #         # 提取类别和置信度
        #         conf = box.conf[0].item()
        #         cls = int(box.cls[0].item())
        #         label = f"{model.names[cls]} {conf:.2f}"
        #
        #         # 绘制矩形和标签
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tuple_keypoint = []
        for result in results:
            xy = result.keypoints.xy  # x and y coordinates
            # print(xy)
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)
            # print(result.keypoints)

            boxes = result.boxes.xyxy

            # print(boxes)
            if xy is not None and boxes is not None:
                # print(xy.cpu().numpy()[0])
                for i in range(len(xy)):
                    # print(xy.cpu().numpy()[0][0])
                    for j in range(len(xy[i])):
                        x = int(xy[i][j][0])
                        y = int(xy[i][j][1])
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, str(keynamedcit[j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
                        cv2.imshow('YOLOv8 Detection', frame)
                        tuple_keypoint.append((x,y))

                for box in boxes:
                    tuple_keypoint.append(box.cpu().numpy().tolist())

        frameid_keypoint[k] = tuple_keypoint
        print("长度：",len(tuple_keypoint),tuple_keypoint)
            # frame = result.plot()
            # print(result.to_json())
        # first_frame.append(frameid_keypoint)
        k+=1

            # 计算FPS
        end_time = time.time()
        # fps = 1 / (end_time - start_time)

        # 在帧上显示FPS
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示结果
        # cv2.imshow('YOLOv8 Detection', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if(frameid_keypoint):
        frameid_keypoint = optimize_keypoints(frameid_keypoint)


    with open(r"H:\workspace\keypoints_diff\testvideos\result20250411094252-test.json", "w", encoding="utf-8") as f:
            json.dump(frameid_keypoint, f, ensure_ascii=False)   #indent=4,

    print(frameid_keypoint)
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()




