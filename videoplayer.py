#coding:utf-8
from PyQt5.QtCore import pyqtSignal, QRect, QSize, Qt
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5.QtWidgets import QWidget
import cv2

# 视频播放控件
class VideoPlayer(QWidget):
    my_signal = pyqtSignal(dict)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.frame=[]
        self.frameid=0
        pass

    def calculate_scaled_rect(self, source_size: QSize, target_size: QSize) -> QRect:
        """计算保持长宽比的绘制区域"""
        source_ratio = source_size.width() / source_size.height()
        target_ratio = target_size.width() / target_size.height()

        if source_ratio > target_ratio:
            # 以宽度为基准缩放
            scaled_h = int(target_size.width() / source_ratio)
            return QRect(0, (target_size.height() - scaled_h) // 2,
                         target_size.width(), scaled_h)
        else:
            # 以高度为基准缩放
            scaled_w = int(target_size.height() * source_ratio)
            return QRect((target_size.width() - scaled_w) // 2, 0,
                         scaled_w, target_size.height())

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        if len(self.frame)>0:
            # tmpframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
            # QImg = QImage(tmpframe.data, tmpframe.shape[1], tmpframe.shape[0], tmpframe.shape[1]*tmpframe.shape[2], QImage.Format_RGB888)
            # painter.drawImage(QRect(0,0,self.width()-1,self.height()-1), QImg)

            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]

            # 计算等比例缩放后的尺寸
            scaled_rect = self.calculate_scaled_rect(QSize(w, h), self.size())

            # 创建QImage并绘制
            qimage = QImage(
                rgb_frame.data,
                w, h,
                rgb_frame.strides[0],  # 自动计算步长
                QImage.Format_RGB888
            )
            painter.drawImage(scaled_rect, qimage)

            # 绘制边框（保留1像素内边距）
        border_rect = QRect(0, 0, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawRect(border_rect)

        # painter.drawRect(0,0,self.width()-1,self.height()-1)
        painter.end()

    def getmat(self,data):
        dictdata = data
        if(len(dictdata)>0):
            self.frame = list(dictdata.values())[0]
            self.frameid = list(dictdata.keys())[0]
            self.update()
