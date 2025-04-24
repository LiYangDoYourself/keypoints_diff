# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/24 14:27
import sys
import json
from vosk import Model, KaldiRecognizer
import pyaudio
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

class VoiceSignals(QObject):
    startRecording = pyqtSignal()
    stopRecording = pyqtSignal()

class VoskVoiceWorker(QRunnable):
    def __init__(self, model_path="vosk-model-small-cn-0.22"):
        super().__init__()
        self.signals = VoiceSignals()
        self.model_path = model_path
        self.running = True

    @pyqtSlot()
    def run(self):
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, 16000)
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        print("🎤 离线语音识别已启动，请说：‘开始录制’或‘结束录制’")
        while self.running:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                print("识别结果：", text)

                if "开始录制" in text:
                    print("发出开始录制信号")
                    self.signals.startRecording.emit()
                elif "结束录制" in text:
                    print("发出停止录制信号")
                    self.signals.stopRecording.emit()

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.running = False
        self.quit()
        # self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 离线语音控制演示")

        self.label = QLabel("语音识别中... 说‘开始录制’或‘结束录制’")
        self.stop_btn = QPushButton("停止监听")
        self.stop_btn.clicked.connect(self.stop_listen)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.threadpool = QThreadPool()
        self.voice_worker = VoskVoiceWorker()
        self.voice_worker.signals.startRecording.connect(self.on_start_recording)
        self.voice_worker.signals.stopRecording.connect(self.on_stop_recording)
        self.threadpool.start(self.voice_worker)

    def on_start_recording(self):
        self.label.setText("✅ 收到『开始录制』")

    def on_stop_recording(self):
        self.label.setText("🛑 收到『结束录制』")

    def stop_listen(self):
        self.voice_worker.stop()
        self.label.setText("🎙️ 已停止语音监听")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

