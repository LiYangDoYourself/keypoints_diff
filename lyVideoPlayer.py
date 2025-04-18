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

#取视频流的地址
class lyVideoStreamThread(QThread):
    matdict_signal = pyqtSignal(dict)

    startrecord_signal = pyqtSignal(dict)
    stoprecord_signal = pyqtSignal()
    widthheight_signal = pyqtSignal(list)

    recordtime_singal = pyqtSignal(str)


    def __init__(self):
        super().__init__()
        self._running = False
        self._paused = False  # 新增暂停状态标志

        self._timeflag=0
        self._flag_startrecord=False
        self._flag_stoprecord=False

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
            if self.path.endswith(".mp4"):
                flag_mp4 = True

            self._timeflag=0   # 用来显示录制了多少秒

            self.start_time=0
            self.end_time=0

            self._running = True # 用来控制线程的运行状态
            # 获取视频帧率
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # 把视频的宽高发到AI处理线程中
            self.widthheight_signal.emit([width,height])

            i = 1

            while self._running:
                if not self._paused:  # 非暂停状态才处理帧
                    ret, frame = self.cap.read()
                    if ret:
                        self.matdict_signal.emit({i:frame})

                        # 是真的就开启录制 ，并处AI处理
                        if(self._flag_startrecord):
                            if self._timeflag==0:
                                self.start_time = time.time()
                                self._timeflag+=1

                            self.startrecord_signal.emit({i:frame})
                            self.end_time = time.time()
                            self.recordtime_singal.emit(self.format_time(self.end_time-self.start_time))

                        if flag_mp4:
                            time.sleep(1/fps)

                        i += 1
                    else:
                        break
                else:
                     time.sleep(1) # 暂停时降低CPU占用
        except Exception as e:
            print(e)

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

#这是一个视频播放控件
class lyVideoPlayer(QWidget):
    my_signal = pyqtSignal(dict)
    uuid_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.path = "rtsp://admin:798446835qqcom@192.168.1.64:554/h264/ch1/main/av_stream"

        self.videostream_thread = lyVideoStreamThread()
        self.videoframeai_thread = AIFrameThread()
        self.flag_startai = False
        self.frame = []
        self.frameid = 0

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

        #发送视频帧
        self.videostream_thread.startrecord_signal.connect(self.videoframeai_thread.putframequeue)

        # 发出录制视频的uuid
        self.videoframeai_thread.uuid_signal.connect(self.returnuuid)

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

    def setrtsp(self,path):
        self.path = path

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

    def next_frame(self,index):
        pass

    def prev_frame(self,index):
        pass

    def returnuuid(self,tmpuuid):
        self.uuid_signal.emit(tmpuuid)

    def setsavepath(self,path):
        self.videoframeai_thread.setsavepath(path)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = lyVideoPlayer()
    window.show()
    sys.exit(app.exec_())