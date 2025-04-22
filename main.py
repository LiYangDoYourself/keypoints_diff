import json
import os.path
import sys
import configparser
import time
from datetime import datetime
import queue

import uuid
import cv2
from PyQt5.QtCore import QTimer, QDateTime, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QDesktopWidget
from PyQt5.uic import loadUi
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQueryModel, QSqlQuery
from PyQt5.QtWidgets import QTableView, QMessageBox

from lyVideoPlayer import *

from lyDataBase import DBWidget

from lyDwt import DTW

# from ultralytics import YOLO
# model_detect = YOLO("best.pt")   # yolov8-x6-pose.pt
#
# keynamedcit={
#     0: "nose",          # 鼻子
#     1: "lefteye",       # 左眼
#     2: "righteye",      # 右眼
#     3: "leftrear",      # 左耳
#     4: "rightrear",     # 右耳
#     5: "leftshoulder",  # 左肩
#     6: "rightshoulder", # 右肩
#     7: "leftelbow",     # 左肘
#     8: "rightelbow",    # 右肘
#     9: "leftwrist",     # 左腕
#     10: "rightwrist",   # 右腕
#     11: "lefthip",      # 左髋
#     12: "righthip",     # 右髋
#     13: "leftknee",     # 左膝
#     14: "rightknee",    # 右膝
#     15: "leftankle",    # 左踝
#     16: "rightankle"    # 右踝
# }


# def getuuid():
#     # 生成并去除连字符
#     quuid = str(uuid.uuid4()).replace("-", "")
#     return  quuid
#
#
# class AIFrameThread(QThread):
#     uuid_signal = pyqtSignal(str)
#     def __init__(self):
#         super().__init__()
#
#         self._width = 1920
#         self._height = 1080
#         self._running = False
#         self._save = r"./testvideos/"
#         self.queue_frame = queue.Queue(maxsize=7500)  # 1*25*60*5  5分钟
#
#     def setwidthheight(self,widthheight):
#         print("widthheight:",widthheight)
#         self._width = int(widthheight[0])
#         self._height = int(widthheight[1])
#
#     def putframequeue(self,matdict):
#         if (len(matdict) > 0):
#             frame = list(matdict.values())[0]
#             frameid = list(matdict.keys())[0]
#             self.queue_frame.put(frame)
#
#     def stop(self):
#         self._running = False
#         # self.quit()
#         # self.wait()
#
#
#
#     def run(self) -> None:
#         try:
#             self._running = True
#             onlyuuid = getuuid()
#
#             videopath = self._save+onlyuuid+".mp4"
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v') #XVID avi
#             out = cv2.VideoWriter(
#                 videopath,
#                 fourcc,
#                 30.0,  # 帧率
#                 (self._width, self._height)  # 分辨率
#             )
#             frameid_keypoint = {}
#             k=0
#             while True:
#                 if not self._running and self.queue_frame.empty():
#                     print("3333333333")
#                     break
#                 if not self.queue_frame.empty():
#                       frame = self.queue_frame.get()
#                       results = model_detect(frame, conf=0.5, imgsz=1280)
#                       tuple_keypoint = []
#                       for result in results:
#                           xy = result.keypoints.xy  # x and y coordinates
#                           # print(xy)
#                           # xyn = result.keypoints.xyn  # normalized
#                           # kpts = result.keypoints.data  # x, y, visibility (if available)
#                           # print(result.keypoints)
#
#                           boxes = result.boxes.xyxy
#
#                           # print(boxes)
#                           if xy is not None and boxes is not None:
#                               # print(xy.cpu().numpy()[0])
#                               for i in range(len(xy)):
#                                   # print(xy.cpu().numpy()[0][0])
#                                   for j in range(len(xy[i])):
#                                       x = int(xy[i][j][0])
#                                       y = int(xy[i][j][1])
#                                       cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#                                       # cv2.putText(frame, str(keynamedcit[j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                       #             (0, 0, 255), 2)
#                                       # cv2.imshow('YOLOv8 Detection', frame)
#                                       tuple_keypoint.append((x, y))
#
#                               for box in boxes:
#                                   tuple_keypoint.append(box.cpu().numpy().tolist())
#
#                       frameid_keypoint[k] = tuple_keypoint
#                       print("长度：", len(tuple_keypoint), tuple_keypoint)
#                       # frame = result.plot()
#                       # print(result.to_json())
#                       # first_frame.append(frameid_keypoint)
#                       k += 1
#                       out.write(frame)
#
#             # 获取视频帧率
#             out.release()
#             self.queue_frame.queue.clear()
#
#             jsonpath = self._save+onlyuuid+".json"
#             with open(jsonpath, "w", encoding="utf-8") as f:
#                 json.dump(frameid_keypoint, f, indent=4, ensure_ascii=False)
#
#
#             if os.path.exists(jsonpath) and os.path.exists(videopath):
#                 self.uuid_signal.emit(onlyuuid)
#
#             print(f"视频已停止")
#         except Exception as e:
#             print(e)

#取视频流的地址
class VideoStreamThread(QThread):
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
                QMessageBox.information(self, "提示", "视频无法连接！")
                return

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

                        i += 1
                    else:
                        break
                else:
                    break  # 暂停时降低CPU占用
        except Exception as e:
            print(e)


        self.stoprecord_signal.emit()

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
        # self.wait()
        print(f"视频已停止")

    def setstartrecord(self):
        self._flag_startrecord=True
        self._flag_stoprecord=False

    def setstoprecord(self):
        self._flag_startrecord=False
        self._flag_stoprecord=True
        self._timeflag == 0  #计算时间
        self.start_time = 0
        self.end_time = 0


class MainWindow(QMainWindow):
    startrecord_signal = pyqtSignal()
    stoprecord_signal = pyqtSignal()

    startcompare_signal = pyqtSignal(list)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.screen_rect = QApplication.primaryScreen().geometry()
        self.statusBar().show()

        self.current_page = 1
        self.total_pages = 0
        self.page_size = 50
        self.stream_thread = VideoStreamThread()
        self.frameAI_thread = AIFrameThread()

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # 创建页面容器
        self.stacked_pages = QStackedWidget()
        # self.setCentralWidget(self.stacked_pages)

        self.main_page_obj = QWidget()
        self.config_page_obj = QWidget()
        self.main_return_obj = QWidget()
        self.videorecord_page_obj = QWidget()
        self.videocompare_page_obj = QWidget()
        self.history_page_obj = QWidget()


        self.lyVideoPlayer_obj1 = lyVideoPlayer()
        self.lyVideoPlayer_obj2 = lyVideoPlayer()
        self.lyVideoPlayer_obj3 = lyVideoPlayer()

        # 数据库页面
        self.database_page_obj = DBWidget()

        # dwt对象
        self.dtw_obj = DTW()

        loadUi('main_page.ui', self.main_page_obj)
        loadUi('config_page.ui', self.config_page_obj)
        loadUi('main_return.ui',self.main_return_obj)
        loadUi('videorecord_page.ui',self.videorecord_page_obj)
        loadUi('videocompare_page.ui',self.videocompare_page_obj)
        loadUi('history_page.ui',self.history_page_obj)





        # 添加到容器
        self.stacked_pages.addWidget(self.main_page_obj)   # 0 主页
        self.stacked_pages.addWidget(self.config_page_obj) # 1  配置页
        self.stacked_pages.addWidget(self.videorecord_page_obj) # 2 动作录制
        self.stacked_pages.addWidget(self.videocompare_page_obj) # 3 动作对比录制页
        self.stacked_pages.addWidget(self.history_page_obj) #4 历史动作

        self.main_layout.addWidget(self.main_return_obj,1)
        self.main_layout.addWidget(self.stacked_pages,15)

        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 默认显示首页
        self.stacked_pages.setCurrentIndex(0)
        self.init_func()
        self.init_slot()

    def init_func(self):
        # 刷新时间
        self.set_main_ui_time()
        # 读取配置
        self.readconfigfile()

        #读取数据库
        self.readdbfile()




    def init_slot(self):
        # 显示配置页
        self.main_page_obj.toolButton_config.clicked.connect(self.transfer_page)
        # 返回主页
        self.main_return_obj.pushButton_config_return.clicked.connect(self.transfer_page)
        #保存配置
        self.config_page_obj.pushButton_config_save.clicked.connect(self.saveconfigfile)

        #切换到action录制
        self.main_page_obj.toolButton_action.clicked.connect(self.transfer_page)

        # 刷新显示tableview
        self.videorecord_page_obj.pushButton_refresh.clicked.connect(lambda:self.videorecord_page_obj.tableView.model().select())

        #删除某一行数据
        self.videorecord_page_obj.pushButton_delete.clicked.connect(self.delete_selected_rows)

        # 选择行并显示
        self.videorecord_page_obj.tableView.doubleClicked.connect(self.show_selected_row)

        # 上一页 下一页
        self.videorecord_page_obj.pushButton_9.clicked.connect(self.go_to_first_page)
        self.videorecord_page_obj.pushButton_7.clicked.connect(self.go_to_prev_page)
        self.videorecord_page_obj.pushButton_8.clicked.connect(self.go_to_next_page)
        self.videorecord_page_obj.pushButton_10.clicked.connect(self.go_to_last_page)

        # 连接视频播放信号
        self.stream_thread.matdict_signal.connect(self.videorecord_page_obj.widget_videoboard.getmat)

        # 播放视频
        self.videorecord_page_obj.pushButton_3.clicked.connect(self.start_video_board)
        # 暂停播放
        self.videorecord_page_obj.pushButton_4.clicked.connect(self.pause_video_board)

        # 开始录制视频
        self.videorecord_page_obj.startrecord_pushButton.clicked.connect(self.start_video_record)
        self.stream_thread.startrecord_signal.connect(self.frameAI_thread.putframequeue)
        # 停止录制
        self.videorecord_page_obj.stoprecord_pushButton.clicked.connect(self.stop_video_record)

        # 视频完毕发送提醒
        self.frameAI_thread.finished.connect(self.finish_video_record)

        # 发送视频长宽
        self.stream_thread.widthheight_signal.connect(self.frameAI_thread.setwidthheight)

        # 发送uuid去前端
        self.frameAI_thread.uuid_signal.connect(self.success_mp4_json)

        # 录制时间显示
        self.stream_thread.recordtime_singal.connect(lambda  timestr:self.videorecord_page_obj.time_label.setText("录制时间"+timestr))

        self.videorecord_page_obj.pushButton_compare.clicked.connect(self.compare_video)

        # 开始测评页面
        self.main_page_obj.toolButton_starttest.clicked.connect(self.transfer_page)

        #设置背景颜色
        self.videorecord_page_obj.pushButton_3.clicked.connect(self.setbtn_color)
        self.videorecord_page_obj.pushButton_4.clicked.connect(self.setbtn_color)
        self.videorecord_page_obj.startrecord_pushButton.clicked.connect(self.setbtn_color)
        self.videorecord_page_obj.stoprecord_pushButton.clicked.connect(self.setbtn_color)
        self.main_return_obj.pushButton_config_return.clicked.connect(self.setbtn_color)

        # 循环播放视频录入进去
        layout_1 = QVBoxLayout()
        layout_1.addWidget(self.lyVideoPlayer_obj1)
        self.videocompare_page_obj.widget.setLayout(layout_1)

        layout_2 = QVBoxLayout()
        layout_2.addWidget(self.lyVideoPlayer_obj2)
        self.videocompare_page_obj.widget_4.setLayout(layout_2)
        # 第二个录制的要开启AI检测的
        self.lyVideoPlayer_obj2.setflag_startai()

        # 历史页面添加 播放器 和 数据库
        layout_3 = QVBoxLayout()
        layout_3.addWidget(self.lyVideoPlayer_obj3)
        self.history_page_obj.widget.setLayout(layout_3)


        layout_4 = QVBoxLayout()
        layout_4.addWidget(self.database_page_obj)
        self.history_page_obj.widget_2.setLayout(layout_4)
        self.database_page_obj.setvideopath(self.configresult['ly']['video_path'])
        self.database_page_obj.double_signal.connect(self.showvideo_selected_row)

        # 跳转到历史
        self.main_page_obj.toolButton_history.clicked.connect(self.transfer_page)


        self.frameAI_thread.setsavepath(self.configresult['ly']['video_path'])
        self.lyVideoPlayer_obj1.setsavepath(self.configresult['ly']['video_path'])
        self.lyVideoPlayer_obj2.setsavepath(self.configresult['ly']['video_path'])

        # 录制完之后传递uuid
        self.lyVideoPlayer_obj2.uuid_signal.connect(self.compare_page_uuid)

        # 对比俩个视频json
        self.videocompare_page_obj.pushButton_compare.clicked.connect(self.compare_Twovideo)

        # 动作页视频暂停
        self.stream_thread.stoprecord_signal.connect(lambda :self.statusBar().showMessage("视频停止"))

        # 成功了就启动对比
        self.startcompare_signal.connect(self.compare_TwovideoResult)


        # 第二个视频+播放按钮
        # self.lyVideoPlayer_obj2.pushButton_3
        #

        # 隐藏导入
        self.videorecord_page_obj.pushButton_import.hide()
        self.videorecord_page_obj.toolButton.hide()

    def transfer_page(self):
        tmpbtn = self.sender()
        if tmpbtn.text() == "返回":
            self.stacked_pages.setCurrentIndex(0)
            self.stream_thread.stop()
            self.frameAI_thread.stop()

        elif tmpbtn.text() == "修改配置":
            self.stacked_pages.setCurrentIndex(1)
        elif tmpbtn.text() == "动作管理":
            self.stacked_pages.setCurrentIndex(2)
        elif tmpbtn.text() == "开始测评":
            self.stacked_pages.setCurrentIndex(3)
            self.stream_thread.stop()
            self.frameAI_thread.stop()
        elif tmpbtn.text()=="历史记录":
            self.stacked_pages.setCurrentIndex(4)




    def setbtn_color(self):

        tmpbtn = self.sender()
        if tmpbtn.text()=="播放":
            self.videorecord_page_obj.pushButton_3.setStyleSheet("background-color: rgb(255,0,0);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.pushButton_4.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.startrecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.stoprecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
        elif tmpbtn.text()=="暂停":
            self.videorecord_page_obj.pushButton_3.setStyleSheet("background-color: rgb(0,0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.pushButton_4.setStyleSheet("background-color: rgb(255, 0, 0);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.startrecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.stoprecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
        elif tmpbtn.text()=="开始录制":
            self.videorecord_page_obj.pushButton_3.setStyleSheet("background-color: rgb(0,0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.pushButton_4.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.startrecord_pushButton.setStyleSheet("background-color: rgb(255, 0, 0);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.stoprecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
        elif tmpbtn.text()=="结束录制":
            self.videorecord_page_obj.pushButton_3.setStyleSheet("background-color: rgb(0,0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.pushButton_4.setStyleSheet("background-color: rgb(0,0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.startrecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.stoprecord_pushButton.setStyleSheet("background-color: rgb(255, 0, 0);font: 18pt 'Agency FB';")
        elif tmpbtn.text()=="返回":
            self.videorecord_page_obj.pushButton_3.setStyleSheet("background-color: rgb(0,0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.pushButton_4.setStyleSheet("background-color: rgb(0, 0,127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.startrecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.stoprecord_pushButton.setStyleSheet("background-color: rgb(0, 0, 127);font: 18pt 'Agency FB';")
            self.videorecord_page_obj.time_label.setText("录制时间")
            self.statusBar().showMessage("主页")
            # 保持窗体最大
            # self.showMaximized()
            # self.setGeometry(0, 0, self.screen_rect.width(), self.screen_rect.height())

    def compare_page_uuid(self,tmpuuid):
        self.videocompare_page_obj.lineEdit_6.setText(tmpuuid)
        self.videocompare_page_obj.lineEdit_8.setText(os.path.join(self.configresult['ly']['video_path'],tmpuuid + ".mp4"))

    def compare_Twovideo(self):
        videopath1 =  self.videocompare_page_obj.lineEdit_4.text()
        videopath2 = self.videocompare_page_obj.lineEdit_8.text()
        if os.path.exists(videopath1) and os.path.exists(videopath2):
            jsonpath1 = videopath1[:-4]+ ".json"
            jsonpath2 =videopath2[:-4]+ ".json"
            if os.path.exists(jsonpath1) and os.path.exists(jsonpath2):

                uuid1 = self.videocompare_page_obj.lineEdit_2.text()
                uuid2 = self.videocompare_page_obj.lineEdit_6.text()
                self.startcompare_signal.emit([uuid1,uuid2])
                self.statusBar().showMessage("视频和数据正在处理中请稍等")

    def compare_TwovideoResult(self,videolist):
        uuid1  = videolist[0]
        uuid2 = videolist[1]

        jsonpath1= os.path.join(self.configresult['ly']['video_path'],uuid1 + ".json")
        jsonpath2= os.path.join(self.configresult['ly']['video_path'],uuid2 + ".json")

        combine_uuid =os.path.join(self.configresult['ly']['video_path'],uuid1+"-"+uuid2+".mp4")

        if os.path.exists(combine_uuid):
            self.statusBar().showMessage("已经存在数据了", 5000)
            self.lyVideoPlayer_obj3.setrtsp(combine_uuid)
            self.stacked_pages.setCurrentIndex(4)
            return

        try:
            self.dtw_obj.setparam(jsonpath1,jsonpath2,combine_uuid)
            self.dtw_obj.readjson()
            self.dtw_obj.dwt_keypoints()
            # self.dtw_obj.finsh_signal.connect()
            self.statusBar().showMessage("数据处理完成",5000)

            action = self.videorecord_page_obj.lineEdit.text() + "," + self.videorecord_page_obj.lineEdit_5.text()
            timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            query = QSqlQuery()
            # 插入数据
            query.prepare("INSERT INTO comparehistory (uuidstandard, uuidcontrast, action,time) VALUES (?, ?, ?, ?)")
            query.addBindValue(uuid1)
            query.addBindValue(uuid2)
            query.addBindValue(action)
            query.addBindValue(timestamp)

            """
            编写dtw 对比俩个视频中关键点的差异给出结论
            """

            if not query.exec():
                print("插入失败:", query.lastError().text())
            else:
                print("插入成功一条对比数据")
                self.statusBar().showMessage("数据插入成功", 5000)

                self.lyVideoPlayer_obj3.setrtsp(combine_uuid)
                ##数据插入成功就会出现这个
                self.stacked_pages.setCurrentIndex(4)

        except Exception as e:
            print(e)


    def set_main_ui_time(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showtime)
        self.timer.start(1000)
    def showtime(self):
        datetime = QDateTime.currentDateTime()
        text = datetime.toString()
        self.main_page_obj.label_time.setText("     " + text)

    def readconfigfile(self):
        """安全读取INI文件并转换为嵌套字典"""
        config = configparser.ConfigParser()
        config.read("./config.ini", encoding='utf-8')

        self.configresult = {}
        for section in config.sections():
            self.configresult[section] = {
                key: config.get(section, key, raw=True)  # 保留原始字符串
                for key in config.options(section)
            }
        print(f"[{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}] 配置文件已加载")

        self.config_page_obj.lineEdit.setText(self.configresult['ly']['rtsp_path'])
        self.config_page_obj.lineEdit_2.setText(self.configresult['ly']['db_path'])
        self.config_page_obj.lineEdit_3.setText(self.configresult['ly']['video_path'])
        self.config_page_obj.lineEdit_4.setText(self.configresult['ly']['model_path'])
        self.config_page_obj.lineEdit_5.setText(self.configresult['ly']['label_score'])
        self.config_page_obj.lineEdit_6.setText(self.configresult['ly']['goal_size'])

        # 这边设置了rtsp地址
        self.stream_thread.setrtsp(self.configresult['ly']['rtsp_path'])

    def saveconfigfile(self):

        config_dict = {
            "ly":{
                "rtsp_path":self.config_page_obj.lineEdit.text(),
                "db_path":self.config_page_obj.lineEdit_2.text(),
                "video_path":self.config_page_obj.lineEdit_3.text(),
                "model_path":self.config_page_obj.lineEdit_4.text(),
                "label_score":self.config_page_obj.lineEdit_5.text(),
                "goal_size":self.config_page_obj.lineEdit_6.text()
                }
        }

        config = configparser.ConfigParser()
        config.read_dict(config_dict)

        with open("./config.ini", 'w', encoding='utf-8') as f:
            config.write(f)

        print(f"[{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}] 配置文件已保存")

    def readdbfile(self):
        if self.configresult is not None:
            db_path = self.configresult['ly']['db_path']
            self.db = QSqlDatabase.addDatabase("QSQLITE")
            self.db.setDatabaseName(db_path)  # 替换为实际路径

            if not self.db.open():
                QMessageBox.critical(None, "错误",
                                     f"[{datetime.now().strftime('%H:%M')}]  数据库连接失败: {self.db.lastError().text()}")

            """创建数据模型并绑定到视图"""
            self.model = QSqlTableModel(self, self.db)
            self.model.setTable("referenceaction")  # 替换为实际表名
            self.model.setFilter(f"1=1 ORDER BY time DESC LIMIT {self.page_size}")  # 关键修改：添加LIMIT子句
            self.model.select()  # 加载数据

            # model = QSqlQueryModel()
            # query = QSqlQuery()
            # query.exec("SELECT * FROM referenceaction LIMIT 2")  # 限制2条
            # model.setQuery(query)


            # 设置表头显示名
            self.model.setHeaderData(0, Qt.Horizontal, "编号")
            self.model.setHeaderData(1, Qt.Horizontal, "动作")
            self.model.setHeaderData(2, Qt.Horizontal, "路径")
            self.model.setHeaderData(3, Qt.Horizontal, "描述")
            self.model.setHeaderData(4, Qt.Horizontal, "时间")

            self.videorecord_page_obj.tableView.setModel(self.model)
            self.videorecord_page_obj.tableView.resizeColumnsToContents()

            #读取有多少数据
            self.calculate_total_pages()

    def show_message(self, text, icon=QMessageBox.Information):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setText(f"{text}\n （{QDateTime.currentDateTime().toString('HH:mm')} ）")
        msg.exec_()

    def delete_selected_rows(self):
        """安全删除选中行（支持数据库和内存模型）"""
        selected = self.videorecord_page_obj.tableView.selectionModel().selectedRows()
        if not selected:
            self.show_message(" 请先选中要删除的行", QMessageBox.Warning)
            return

            # 确认对话框（含当前时间显示）
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除选中的 {len(selected)} 行吗？\n（操作时间：{QDateTime.currentDateTime().toString('HH:mm:ss')} ）",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            model = self.videorecord_page_obj.tableView.model()
            for index in sorted(selected, key=lambda x: x.row(), reverse=True):  # 倒序避免索引错乱
                model.removeRow(index.row())

            if isinstance(model, QSqlTableModel):
                model.submitAll()  # 数据库提交
            self.show_message(" 删除成功", QMessageBox.Information)
            self.calculate_total_pages()

    def show_selected_row(self,index):
        """将选中行数据填充到对应LineEdit"""
        model = self.videorecord_page_obj.tableView.model()
        row = index.row()

        # 定义表格列与LineEdit的映射关系
        column_mapping = {
            0: self.videorecord_page_obj.lineEdit_5,  # 第0列 -> ID输入框
            1: self.videorecord_page_obj.lineEdit,  # 第1列 -> 行为
            2: self.videorecord_page_obj.lineEdit_4,  # 第2列 -> 视频路径
            3: self.videorecord_page_obj.lineEdit_2 #  第3列-> 描述
        }

        # 遍历映射关系填充数据setdbinfo
        for col, widget in column_mapping.items():
            widget.setText(str(model.data(model.index(row, col))))

        # 状态栏提示（含当前时间）
        self.statusBar().showMessage(
            f"数据已加载 | 更新时间：{QDateTime.currentDateTime().toString('HH:mm:ss')} ",
            3000
        )



        videopath = self.videorecord_page_obj.lineEdit_4.text()
        cap = cv2.VideoCapture(videopath)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.namedWindow("video", 0)
            while True:
                if cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1:
                    break
                ret,frame = cap.read()
                if(ret):
                    cv2.imshow("video",frame)
                    cv2.waitKey(int(1000/fps))
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
    def calculate_total_pages(self):
        query = QSqlQuery("SELECT COUNT(*) FROM referenceaction")
        if query.next():
            total_items = query.value(0)
            self.total_pages = max(1, (total_items + self.page_size - 1) // self.page_size)


    def load_data(self):

        # 正确创建QSqlQuery对象（当前时间：17:42 农历三月十八）
        query = QSqlQuery(self.db)  # 需传入数据库连接
        offset = (self.current_page - 1) * self.page_size
        sql = f"SELECT * FROM referenceaction LIMIT {self.page_size}  OFFSET {offset}"

        if not query.exec(sql):
            error = query.lastError().text()
            QMessageBox.critical(self, "错误",
                                 f"查询执行失败\n时间：{QDateTime.currentDateTime().toString('HH:mm:ss')}\n 错误：{error}")
            return

            # 设置查询模型
        model = self.videorecord_page_obj.tableView.model()
        if isinstance(model, QSqlQueryModel):
            model.setQuery(query)  # 传入QSqlQuery对象而非字符串
            self.update_pagination_status()

    def update_pagination_status(self):
        """更新分页状态"""
        self.videorecord_page_obj.lineEdit_3.setText(f" 第{self.current_page} 页/共{self.total_pages} 页")
        self.videorecord_page_obj.pushButton_7.setEnabled(self.current_page > 1)  # 上一页
        self.videorecord_page_obj.pushButton_8.setEnabled(self.current_page < self.total_pages) # 下一页

    def go_to_first_page(self):
        self.current_page = 1
        self.load_data()

    def go_to_prev_page(self):
        self.current_page = max(1, self.current_page - 1)
        self.load_data()

    def go_to_next_page(self):
        self.current_page = min(self.total_pages, self.current_page + 1)
        self.load_data()

    def go_to_last_page(self):
        self.current_page = self.total_pages
        self.load_data()

    def start_video_board(self):
        if not self.stream_thread.isRunning():
            self.stream_thread.start()
            self.statusBar().showMessage("视频正在播放ing")
    def pause_video_board(self):
        """切换暂停/恢复状态"""
        if self.stream_thread.isRunning():
            self.stream_thread.pause()
            self.stream_thread.stop()
            self.statusBar().showMessage("视频已停止")

    def start_video_record(self):
        if not self.frameAI_thread.isRunning() and self.stream_thread.isRunning():
            if self.videorecord_page_obj.startrecord_pushButton.isEnabled():
                self.videorecord_page_obj.startrecord_pushButton.setEnabled(False)
                self.stream_thread.setstartrecord()
                print("启动视频录制和AI处理")
                self.frameAI_thread.start()
                self.statusBar().showMessage("视频正在录制中。。。")


    def stop_video_record(self):
        if not self.videorecord_page_obj.startrecord_pushButton.isEnabled():
            self.videorecord_page_obj.startrecord_pushButton.setEnabled(True)
            # 先停止送帧数据到AI线程
            print("1111111")
            self.stream_thread.setstoprecord()
            print("222222")
            # if(self.frameAI_thread.isRunning()):
            self.frameAI_thread.stop()
            self.statusBar().showMessage("等待视频AI处理，请稍等。。。")

    def finish_video_record(self):
        self.statusBar().showMessage("视频处理完毕")
        if not self.videorecord_page_obj.startrecord_pushButton.isEnabled():
            self.videorecord_page_obj.startrecord_pushButton.setEnabled(True)

    def success_mp4_json(self,tmpuuid):

         self.videorecord_page_obj.lineEdit_5.setText(tmpuuid)
         self.videorecord_page_obj.lineEdit_4.setText(tmpuuid + ".mp4")

         # 输出示例：2025-04-16 15:36:22
         timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
         videopath = os.path.join(self.configresult["ly"]["video_path"],tmpuuid + ".mp4")
         onlyuuid = tmpuuid
         describle = self.videorecord_page_obj.lineEdit_2.text()
         action = self.videorecord_page_obj.lineEdit.text()

         sql = "INSERT INTO referenceaction VALUES (?,?,?,?,?)"
         query = QSqlQuery()
         query.prepare(sql)

         # 绑定参数（含时间戳）
         params = (
             onlyuuid,
             action,
             videopath,
             describle,
             timestamp  # ISO 8601格式
         )
         for i, value in enumerate(params):
             query.addBindValue(value)

             # 执行并验证
         if not query.exec():
             print("Error:", query.lastError().text())
         else:

             # 更新计算总页数
             self.calculate_total_pages()

             # 刷新一下界面
             self.videorecord_page_obj.tableView.model().select()

    def compare_video(self):

        self.videocompare_page_obj.lineEdit.setText(self.videorecord_page_obj.lineEdit.text())
        self.videocompare_page_obj.lineEdit_2.setText(self.videorecord_page_obj.lineEdit_5.text())
        self.videocompare_page_obj.lineEdit_3.setText(self.videorecord_page_obj.lineEdit_2.text())
        self.videocompare_page_obj.lineEdit_4.setText(self.videorecord_page_obj.lineEdit_4.text())
        self.lyVideoPlayer_obj1.setrtsp(self.videocompare_page_obj.lineEdit_4.text())

        self.lyVideoPlayer_obj2.setrtsp(self.configresult['ly']['rtsp_path'])

        self.stacked_pages.setCurrentIndex(3)
        self.statusBar().showMessage("开始测评")

    def showvideo_selected_row(self,uuidlist):
        # boradvideo,uuid1,uuid2 合并之后视频地址
        self.lyVideoPlayer_obj3.setrtsp(uuidlist[0])
        self.lyVideoPlayer_obj3.start_videoborad()

    def closeEvent(self, event):
            self.stream_thread.stop()
            self.frameAI_thread.stop()
            event.accept()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()
    app.exec_()