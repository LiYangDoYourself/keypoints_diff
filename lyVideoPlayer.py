# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/17 10:21
import time

from PyQt5.QtCore import pyqtSignal, QRect, QSize, Qt, QThread
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5.QtWidgets import QWidget
import cv2
from PyQt5.uic import loadUi

from lyAiDetect import *

import numpy as np


#取视频流的地址
class lyVideoStreamThread(QThread):
    matdict_signal = pyqtSignal(dict)

    startrecord_signal = pyqtSignal(dict)
    stoprecord_signal = pyqtSignal()
    widthheight_signal = pyqtSignal(list)

    recordtime_singal = pyqtSignal(str)

    sendframeindex_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._running = False
        self._paused = False  # 新增暂停状态标志

        self._timeflag=0
        self._flag_startrecord=False
        self._flag_stoprecord=False
        self._flag_loop=False

    def setrtsp(self,path):
        self.path = path

    def format_time(self,seconds):
        """将秒数转换为 HH:MM:SS 格式"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                return

            flag_mp4=False
            if self.path.endswith(".mp4") or self.path.endswith(".avi"):
                flag_mp4 = True

            self._timeflag=0   # 用来显示录制了多少秒

            self.start_time=0
            self.end_time=0

            self._running = True # 用来控制线程的运行状态
            # 获取视频帧率
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 把视频的宽高发到AI处理线程中
            self.widthheight_signal.emit([width,height])



            self.frame_index = 0

            while self._running:
                if not self._paused:  # 非暂停状态才处理帧
                    ret, frame = self.cap.read()

                    if  self._flag_loop and  not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_index = 0
                        continue

                    if ret:
                        self.matdict_signal.emit({self.frame_index:frame})

                        # 是真的就开启录制 ，并处AI处理
                        if(self._flag_startrecord):
                            if self._timeflag==0:
                                self.start_time = time.time()
                                self._timeflag+=1

                            self.startrecord_signal.emit({self.frame_index:frame})
                            self.end_time = time.time()
                            self.recordtime_singal.emit(self.format_time(self.end_time-self.start_time))

                        if flag_mp4:
                            time.sleep(1/fps)

                        self.frame_index += 1
                    else:
                        break
                else:
                     time.sleep(1) # 暂停时降低CPU占用
        except Exception as e:
            print(e)

        # ✅ 上一帧控制
    def prev_frame(self):
        if not self._paused or not self.cap or not self.cap.isOpened():
            print("未处于暂停状态或视频无效")
            return

        self.frame_index = max(0, self.frame_index - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.sendframeindex_signal.emit(self.frame_index)
            self.matdict_signal.emit({self.frame_index: frame})
            print(f"跳转到上一帧：{self.frame_index}")
        else:
            print("无法读取上一帧")

        # ✅ 下一帧控制
    def next_frame(self):
        if not self._paused or not self.cap or not self.cap.isOpened():
            print("未处于暂停状态或视频无效")
            return

        if self.frame_index + 1 >= self.total_frames:
            print("已经是最后一帧")
            return

        self.frame_index += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.sendframeindex_signal.emit(self.frame_index)
            self.matdict_signal.emit({self.frame_index: frame})
            print(f"跳转到下一帧：{self.frame_index}")
        else:
            print("无法读取下一帧")

    def pause(self):
        """暂停视频流（保持线程运行但不发送帧）"""
        self._paused = True
        print(f"视频已暂停")

    def resume(self):
        """恢复视频流"""
        self._paused = False
        print(f"视频已恢复")

    def stop(self):
        """完全停止线程"""
        self._running = False
        self._paused = False
        self.wait()
        print(f"视频已停止")

    def setstartrecord(self):
        self._flag_startrecord=True
        self._flag_stoprecord=False
        print("启动录制")

    def setstoprecord(self):
        self._flag_startrecord=False
        self._flag_stoprecord=True
        self._timeflag == 0  #计算时间
        self.start_time = 0
        self.end_time = 0
        print("停止录制")

    def setflagloop(self):
        self._flag_loop=True


#这是一个视频播放控件
class lyVideoPlayer(QWidget):
    my_signal = pyqtSignal(dict)
    uuid_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.path = "./testvideos/pose_video.mp4"
        self.jsonpath = None
        self.storepath = "./testvideos"
        self.videostream_thread = lyVideoStreamThread()
        self.videoframeai_thread = AIFrameThread()
        self.flag_startai = False  # 控制AI录制和停止录制

        self.flag_loop = False  # 是否循环播放
        self.frame = []
        self.frameid = 0
        self.flag_nextpre=False   # 控制上一帧下一帧

        self.listdata = []
        self.id1json_data = []
        self.id2json_data = []
        loadUi('videoborad_page.ui',self)

        self.init_slot()

    def init_slot(self):

        # 播放视频
        self.pushButton.clicked.connect(self.start_videoborad)

        # 暂停播放
        self.pushButton_2.clicked.connect(self.pause_videoborad)

        # 恢复播放
        self.pushButton_9.clicked.connect(self.resume_videoborad)

        # 停止播放
        self.pushButton_8.clicked.connect(self.stop_videoborad)

        # 启动录制
        self.pushButton_3.clicked.connect(self.start_videorecord)

        # 停止录制
        self.pushButton_4.clicked.connect(self.stop_videorecord)

        # 把视频发送到提升的类上
        self.videostream_thread.matdict_signal.connect(self.widget_video.getmat)

        # 设置视频的宽高
        self.videostream_thread.widthheight_signal.connect(self.videoframeai_thread.setwidthheight)

        # 发送视频帧
        self.videostream_thread.startrecord_signal.connect(self.videoframeai_thread.putframequeue)

        # 发出录制视频的uuid
        self.videoframeai_thread.uuid_signal.connect(self.returnuuid)

        # 上一帧 下一帧
        self.pushButton_5.clicked.connect(self.prev_frame)
        self.pushButton_6.clicked.connect(self.next_frame)

        # 设置相关的参数和需求说明

        self.videostream_thread.sendframeindex_signal.connect(self.get_frameid_deal)

        # 按钮列表，方便统一管理
        self.buttons = [
            self.pushButton,
            self.pushButton_2,
            self.pushButton_3,
            self.pushButton_4,
            self.pushButton_5,
            self.pushButton_6,
            self.pushButton_7,
            self.pushButton_8,
            self.pushButton_9
        ]

        # 为每个按钮绑定点击事件
        for btn in self.buttons:
            btn.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        # 获取点击的按钮
        clicked_button = self.sender()
        # 更新按钮样式
        self.update_button_styles(clicked_button)

    def update_button_styles(self, active_button):
        # 定义样式
        active_style = "background-color: rgb(255,0,0); font: 18pt 'Agency FB';"
        inactive_style = "background-color: rgb(0,0,127); font: 18pt 'Agency FB';"

        # 遍历所有按钮，设置样式
        for btn in self.buttons:
            if btn == active_button:
                btn.setStyleSheet(active_style)
            else:
                btn.setStyleSheet(inactive_style)

    def start_videoborad(self):

        if not self.videostream_thread.isRunning():
            self.videostream_thread.setrtsp(self.path)
            self.videostream_thread.start()

    def pause_videoborad(self):

        if self.videostream_thread.isRunning():
            self.videostream_thread.pause()

    def resume_videoborad(self):

        if self.videostream_thread.isRunning():
            self.videostream_thread.resume()
    def stop_videoborad(self):

        if self.videostream_thread.isRunning():
            self.videostream_thread.stop()

    def setflag_startai(self):
        self.flag_startai = True

    def start_videorecord(self):
        if(self.videostream_thread.isRunning() and self.flag_startai):
            self.videostream_thread.setstartrecord()
            if not self.videoframeai_thread.isRunning():
                self.videoframeai_thread.start()
    def stop_videorecord(self):

        if(self.videostream_thread.isRunning()):
            self.videostream_thread.setstoprecord()
            self.videoframeai_thread.stop()

    def next_frame(self):

        if(self.videostream_thread.isRunning() and self.flag_nextpre):
            self.videostream_thread.next_frame()
    def prev_frame(self):
        if (self.videostream_thread.isRunning() and self.flag_nextpre):
            self.videostream_thread.prev_frame()

    def returnuuid(self,tmpuuid):
        self.uuid_signal.emit(tmpuuid)

    def setsavepath(self,path):
        self.videoframeai_thread.setsavepath(path)

    def setrtsp(self, path):
        self.path = path
    def setcombinejson(self, jsonpath, uuid1, uuid2):
        self.jsonpath = jsonpath  # xxx/xxx/xx.json
        self.uuid1 = uuid1  # xxx1
        self.uuid2 = uuid2  # xxx2

    #这个要设置的
    def setnextpreflag(self):
        self.flag_nextpre = True

    def setflagloop(self):
        self.videostream_thread.setflagloop()
    def setstorepath(self, storepath):
        self.storepath = storepath
    def readjsondata(self):

        try:

            if(os.path.exists(self.jsonpath)):
                with open(self.jsonpath, "r", encoding="utf-8") as f:
                    self.listdata = json.load(f)

            # id1 json路径
            id1json_path = os.path.join(self.storepath, self.uuid1 + ".json")
            id2json_path = os.path.join(self.storepath, self.uuid2 + ".json")
            # 提取关键点数据
            if(os.path.exists(id1json_path) and os.path.exists(id2json_path)):
                with open(id1json_path, 'r') as f:
                    self.id1json_data = json.load(f)

                with open(id2json_path, 'r') as f:
                    self.id2json_data = json.load(f)

        except Exception as e:
            print(e)

    # ========== 角度差异函数 ==========
    def calculate_angle(self,a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    # 角度计算函数（与竖直方向的夹角）
    def calculate_torso_angle(self, neck, hip_center):
        vector = hip_center - neck
        vertical = np.array([0, 1])  # 向下为竖直方向
        cos_theta = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止数值误差
        return np.degrees(angle)

    # ========== 姿态结构归一化对比 ==========
    def normalize_keypoints(self,kp):
        origin = (kp[0]+kp[1])/2  # 肩膀中间
        kp = kp - origin
        scale = np.linalg.norm(kp)
        return kp / (scale + 1e-8)


    #回传相关的对比数据
    def get_frameid_deal(self,frameid):
        try:
            # if os.path.exists(self.jsonpath):
            if (frameid<len(self.listdata)):

                # 提取出来匹配的id1和id2
                frame_id1id2 = self.listdata[frameid]

                # 提取关键点数据   注意这里面的标签是 key "0":data  darashape:[17,2]
                # keypointid1 = self.id1json_data[str(frame_id1id2[0])]
                # keypointid2 = self.id2json_data[str(frame_id1id2[1])]
                #
                # print("keypointid1:",keypointid1)
                # print("keypointid2:",keypointid2)

                # 注意这里要转换成
                kp1 = np.array(self.id1json_data[str(frame_id1id2[0])][0:17])
                kp2 = np.array(self.id2json_data[str(frame_id1id2[1])][0:17])

                print("keypointid1:",kp1)
                print("keypointid2:",kp2)


                # 示例角度对比（左臂、右臂、左腿、右腿）
                angles1 = [
                    self.calculate_angle(kp1[5], kp1[7], kp1[9]),  # 左肩-肘-腕
                    self.calculate_angle(kp1[6], kp1[8], kp1[10]),  # 右肩-肘-腕
                    self.calculate_angle(kp1[11], kp1[13], kp1[15]),  # 左髋-膝-踝
                    self.calculate_angle(kp1[12], kp1[14], kp1[16]),  # 右髋-膝-踝
                    self.calculate_torso_angle((kp1[5]+kp1[6]) / 2,(kp1[11] + kp1[12]) / 2)
                ]
                angles2 = [
                    self.calculate_angle(kp2[5], kp2[7], kp2[9]),
                    self.calculate_angle(kp2[6], kp2[8], kp2[10]),
                    self.calculate_angle(kp2[11], kp2[13], kp2[15]),
                    self.calculate_angle(kp2[12], kp2[14], kp2[16]),
                    self.calculate_torso_angle((kp2[5] + kp2[6]) / 2, (kp2[11] + kp2[12]) / 2)
                ]
                angle_diff = np.abs(np.array(angles1) - np.array(angles2))

                # ========== 欧氏距离 ==========[]
                euclidean_diff = np.linalg.norm(kp1[5:17] - kp2[5:17], axis=1)
                mean_distance = np.mean(euclidean_diff)

                # ========== 姿态结构归一化对比 ==========
                norm1 = self.normalize_keypoints(kp1[5:17])
                norm2 = self.normalize_keypoints(kp2[5:17])
                structure_diff = np.linalg.norm(norm1 - norm2)

                print("整体结构差异:", structure_diff)
                print("角度差（左臂、右臂、左腿、右腿、躯干角度差（脖子→髋部）:", angle_diff)
                print("平均欧氏距离:", mean_distance)
                print("平均角度差:", np.mean(angle_diff))

        except Exception as e:
            print(e)



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = lyVideoPlayer()
    window.show()
    sys.exit(app.exec_())