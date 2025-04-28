# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/27 15:51
import math

from PyQt5.QtChart import QChart, QChartView, QValueAxis, QLineSeries
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QCursor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QToolTip, QDialog




class lyChartView(QWidget):
    def __init__(self,parent=None):

        super().__init__(parent)
        # 1. 初始化图表组件
        self.chart = QChart()
        # self.chart.setTitle(title)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)  # 华为昇腾NPU加速渲染
        self.chart.legend().setAlignment(Qt.AlignRight)


        # 2. 创建视图容器
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

        # 3. 坐标轴配置
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self._setup_axes()

        # 4. 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.chart_view)
        self.setLayout(layout)

        # 5. 数据容器
        self.series_dict = {}  # 存储多组曲线

        self.chart_view.mouseDoubleClickEvent = self.open_large_chart

    def settitlename(self,title):

        self.chart.setTitle(title)
        self.chart_view.repaint()
    def _setup_axes(self):
        """初始化坐标轴参数"""
        self.axis_x.setTitleText("帧数")
        self.axis_x.setRange(0, 300)  # 默认60秒时间轴

        self.axis_x.setTickInterval(50)  # <--- 新增！每隔10帧一个标签
        self.axis_x.setLabelsAngle(-45)

        self.axis_y.setTitleText("角度值")
        self.axis_y.setRange(0, 180)

        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

    def open_large_chart(self, event):

        """双击放大 chart —— 使用复制版"""
        dialog = QDialog(self)
        dialog.setWindowTitle("放大查看")
        dialog.resize(800, 600)

        # 创建新的 chart 和 chartView
        new_chart = QChart()
        new_chart.setTitle(self.chart.title())
        new_chart.setAnimationOptions(QChart.SeriesAnimations)
        new_chart.legend().setAlignment(Qt.AlignRight)

        # 创建新的坐标轴
        axis_x = QValueAxis()
        axis_x.setTitleText(self.axis_x.titleText())
        axis_x.setRange(self.axis_x.min(), self.axis_x.max())
        axis_x.setTickInterval(self.axis_x.tickInterval())
        axis_x.setLabelsAngle(-45)

        axis_y = QValueAxis()
        axis_y.setTitleText(self.axis_y.titleText())
        axis_y.setRange(self.axis_y.min(), self.axis_y.max())

        new_chart.addAxis(axis_x, Qt.AlignBottom)
        new_chart.addAxis(axis_y, Qt.AlignLeft)

        # 复制原有的数据
        for name, series in self.series_dict.items():
            new_series = QLineSeries()
            new_series.setName(name)
            new_series.setColor(series.color())
            for point in series.pointsVector():
                new_series.append(point)
            new_chart.addSeries(new_series)
            new_series.attachAxis(axis_x)
            new_series.attachAxis(axis_y)

            # 悬浮提示也绑定
            new_series.hovered.connect(self.show_tooltip_zoom)

        # 新的 chartView
        chart_view = QChartView(new_chart)
        chart_view.setRenderHint(QPainter.Antialiasing)

        layout = QVBoxLayout()
        layout.addWidget(chart_view)
        dialog.setLayout(layout)

        dialog.exec_()

    def show_tooltip(self, point: QPointF, state: bool):
        """鼠标悬浮在数据点上时，显示提示"""
        if state:  # 只有悬浮到点上时才显示
            # 设置提示内容
            text = f"帧数: {point.x():.0f}\n角度值: {point.y():.1f}"
            # 获取当前鼠标位置
            pos = self.mapFromGlobal(QCursor.pos())
            # 弹出提示框
            QToolTip.showText(QCursor.pos(), text, self.chart_view)

    def show_tooltip_zoom(self, point: QPointF, state: bool):
        """放大窗口里的提示"""
        if state:
            text = f"帧数: {point.x():.0f}\n角度值: {point.y():.1f}"
            QToolTip.showText(QCursor.pos(), text)

    def add_series(self, name, color=Qt.blue):
        """添加新曲线"""
        series = QLineSeries()
        series.setName(name)
        series.setColor(color)

        self.series_dict[name] = series
        self.chart.addSeries(series)
        series.attachAxis(self.axis_x)
        series.attachAxis(self.axis_y)

        # 连接hover信号到槽函数
        series.hovered.connect(self.show_tooltip)

        return series

    def append_data(self, series_name, x, y):
        """动态追加数据点"""
        if series_name in self.series_dict:
            self.series_dict[series_name].append(QPointF(x, y))

            # 自动调整x轴最大值
            if x > self.axis_x.max():
                self.axis_x.setMax(x + 10)  # 给一点缓冲区

            # 自动滚动显示（保留最近60秒）
            # if x > self.axis_x.max():
            #     self.axis_x.setRange(x - 60, x)

    def clear_all(self):
        """清空所有数据"""
        for series in self.series_dict.values():
            series.clear()
        # self.axis_x.setRange(0, 60)



if __name__ == '__main__':

    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    # 初始化
    chart = lyChartView()

    # 添加曲线
    temp_series = chart.add_series("标准动作", Qt.red)
    hum_series = chart.add_series("测试动作", Qt.blue)

    # 模拟数据更新
    for t in range(100):
        chart.append_data("标准动作", t, math.sin(t / 10))
        chart.append_data("测试动作", t, math.cos(t / 8))
        QApplication.processEvents()  # 保持UI响应

    chart.show()
    sys.exit(app.exec_())


