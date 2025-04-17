# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/17 11:55
import json
import os
import time
from datetime import datetime
import queue

import uuid
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from ultralytics import YOLO
model_detect = YOLO("best.pt")   # yolov8-x6-pose.pt

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


def getuuid():
    # 生成并去除连字符
    quuid = str(uuid.uuid4()).replace("-", "")
    return  quuid


class AIFrameThread(QThread):
    uuid_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()

        self._width = 1920
        self._height = 1080
        self._running = False
        self._save = r"./testvideos/"
        self.queue_frame = queue.Queue(maxsize=7500)  # 1*25*60*5  5分钟

    def setwidthheight(self,widthheight):
        print("widthheight:",widthheight)
        self._width = int(widthheight[0])
        self._height = int(widthheight[1])

    def putframequeue(self,matdict):
        if (len(matdict) > 0):
            frame = list(matdict.values())[0]
            frameid = list(matdict.keys())[0]
            self.queue_frame.put(frame)

    def setsavepath(self,path):
        self._save = path

    def stop(self):
        self._running = False
        # self.quit()
        # self.wait()



    def run(self) -> None:
        try:
            self._running = True
            onlyuuid = getuuid()

            videopath = self._save+onlyuuid+".mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') #XVID avi
            out = cv2.VideoWriter(
                videopath,
                fourcc,
                30.0,  # 帧率
                (self._width, self._height)  # 分辨率
            )
            frameid_keypoint = {}
            k=0
            while True:
                if not self._running and self.queue_frame.empty():
                    print("3333333333")
                    break
                if not self.queue_frame.empty():
                      frame = self.queue_frame.get()
                      results = model_detect(frame, conf=0.5, imgsz=1280)
                      tuple_keypoint = []
                      for result in results:
                          xy = result.keypoints.xy  # x and y coordinates
                          # print(xy)
                          # xyn = result.keypoints.xyn  # normalized
                          # kpts = result.keypoints.data  # x, y, visibility (if available)
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
                                      # cv2.putText(frame, str(keynamedcit[j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      #             (0, 0, 255), 2)
                                      # cv2.imshow('YOLOv8 Detection', frame)
                                      tuple_keypoint.append((x, y))

                              for box in boxes:
                                  tuple_keypoint.append(box.cpu().numpy().tolist())

                      frameid_keypoint[k] = tuple_keypoint
                      print("长度：", len(tuple_keypoint), tuple_keypoint)
                      # frame = result.plot()
                      # print(result.to_json())
                      # first_frame.append(frameid_keypoint)
                      k += 1
                      out.write(frame)

            # 获取视频帧率
            out.release()
            self.queue_frame.queue.clear()

            jsonpath = self._save+onlyuuid+".json"
            with open(jsonpath, "w", encoding="utf-8") as f:
                json.dump(frameid_keypoint, f, indent=4, ensure_ascii=False)


            if os.path.exists(jsonpath) and os.path.exists(videopath):
                self.uuid_signal.emit(onlyuuid)

            print(f"视频已停止")
        except Exception as e:
            print(e)