# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/21 16:07
import sqlite3
import sys
import sqlite3
from datetime import datetime

from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableView,
    QSpinBox, QMessageBox
)
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel
from PyQt5.uic import loadUi


# 每页显示条数

class DBWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_page = 0
        self.total_records = 0
        self.total_pages = 0
        self.PAGE_SIZE = 2
        self.db_path = "keypointdb"

        self.current_tablename="comparehistory"

        self.database_page_obj= self
        loadUi('db_page.ui', self.database_page_obj)

        self.init_db()
        self.init_ui()
        self.load_data()


    def setdbinfo(self,dbpath,tablename):

        self.db_path = dbpath
        self.current_tablename = tablename

    def init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        # cursor = self.conn.cursor()
        #
        # # 测试数据插入
        # cursor.execute("SELECT COUNT(*) FROM users")
        # if cursor.fetchone()[0] == 0:
        #     for i in range(100):
        #         cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", (f"User{i}", 20 + i % 10))
        #     self.conn.commit()

        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(self.db_path)  # 替换为实际路径

        if not self.db.open():
            QMessageBox.critical(None, "错误",
                                 f"[{datetime.now().strftime('%H:%M')}]  数据库连接失败: {self.db.lastError().text()}")


    def init_ui(self):
        self.setWindowTitle("数据库分页示例")
        # self.setGeometry(100, 100, 800, 600)
        # self.setFixedSize(800, 600)


        self.table = self.database_page_obj.tableView
        self.model = QSqlTableModel(self, self.db)
        self.model.setTable(self.current_tablename)  # 设置表名字
        self.table.setModel(self.model)


        self.btn_first = self.database_page_obj.pushButton_3
        self.btn_prev = self.database_page_obj.pushButton
        self.btn_next = self.database_page_obj.pushButton_2
        self.btn_last = self.database_page_obj.pushButton_4
        # self.spin_page = QSpinBox()
        # self.spin_page.setMinimum(1)
        self.label_page = self.database_page_obj.label
        self.btn_refresh = self.database_page_obj.pushButton_6
        self.btn_delete = self.database_page_obj.pushButton_5

        # 绑定事件
        self.btn_first.clicked.connect(self.go_first)
        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_next.clicked.connect(self.go_next)
        self.btn_last.clicked.connect(self.go_last)
        # self.btn_go.clicked.connect(self.go_to_page)
        self.btn_refresh.clicked.connect(self.load_data)
        self.btn_delete.clicked.connect(self.delete_selected_rows)

    def load_data(self):
        query = sqlite3.connect(f"{self.db_path}").cursor()
        query.execute(f"SELECT COUNT(*) FROM {self.current_tablename}")
        self.total_records = query.fetchone()[0]
        self.total_pages = (self.total_records - 1) // self.PAGE_SIZE + 1
        # self.spin_page.setMaximum(max(1, self.total_pages))

        offset = self.current_page * self.PAGE_SIZE

        self.model.setFilter(f"1=1 ORDER BY time DESC LIMIT {self.PAGE_SIZE} OFFSET {offset}")
        self.model.select()

        self.model.setHeaderData(0, Qt.Horizontal, "标准动作id")
        self.model.setHeaderData(1, Qt.Horizontal, "对比动作id")
        self.model.setHeaderData(2, Qt.Horizontal, "动作")
        self.model.setHeaderData(3, Qt.Horizontal, "时间")

        self.update_pagination_status()


    def go_first(self):
        self.current_page = 0
        self.load_data()

    def go_prev(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_data()

    def go_next(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_data()

    def go_last(self):
        self.current_page = self.total_pages - 1
        self.load_data()

    def go_to_page(self):
        page = self.spin_page.value() - 1
        if 0 <= page < self.total_pages:
            self.current_page = page
            self.load_data()

    def show_message(self, text, icon=QMessageBox.Information):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setText(f"{text}\n （{QDateTime.currentDateTime().toString('HH:mm')} ）")
        msg.exec_()

    def update_pagination_status(self):
        """更新分页状态"""
        self.label_page.setText(f" 第{self.current_page} 页/共{self.total_pages} 页")
        # self.btn_prev.setEnabled(self.current_page > 1)  # 上一页
        # self.btn_next.setEnabled(self.current_page < self.total_pages) # 下一页
    def delete_selected_rows(self):
        """安全删除选中行（支持数据库和内存模型）"""
        selected = self.table.selectionModel().selectedRows()
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
            model = self.table.model()
            for index in sorted(selected, key=lambda x: x.row(), reverse=True):  # 倒序避免索引错乱
                model.removeRow(index.row())

            if isinstance(model, QSqlTableModel):
                model.submitAll()  # 数据库提交
            self.show_message(" 删除成功", QMessageBox.Information)
            self.load_data()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DBWidget()
    win.show()
    sys.exit(app.exec_())
