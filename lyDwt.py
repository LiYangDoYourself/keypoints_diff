# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/21 18:25
import json



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtw import dtw
from sklearn.metrics.pairwise import cosine_similarity
from calculate_angle import *


from PyQt5.QtCore import QThread, pyqtSignal


class DTW(QThread):
    finsh_signal = pyqtSignal()
    def __init__(self):
        super().__init__()

        self.json1 = None
        self.json2 = None
        self.combine_uuid = None

        self.video_A_raw = []
        self.video_B_raw = []

        self.video_A_rect = []
        self.video_B_rect = []

        self._combine_id=[]

    def getcombineid(self):
        return self._combine_id

    def setparam(self,json1,json2,combine_uuid):
        self.json1 = json1
        self.json2 = json2
        self.combine_uuid = combine_uuid


    def readjson(self):

        self.video_A_raw = []
        self.video_B_raw = []

        self.video_A_rect = []
        self.video_B_rect = []

        self._combine_id = []

        with open(self.json1 , "r",
                  encoding="utf-8") as f:
            data = json.load(f)

            print(data)

            for key, value in data.items():
                if len(value[-1]) == 4:
                    self.video_A_raw.append(value[:-1])
                    self.video_A_rect.append(value[-1])

            # video_A_raw = interpolate_missing_keypoints(video_A_raw)

        with open(self.json2, "r",
                  encoding="utf-8") as f2:
            data = json.load(f2)
            print(data)
            for key, value in data.items():
                if len(value[-1]) == 4:
                    self.video_B_raw.append(value[:-1])
                    self.video_B_rect.append(value[-1])

    def euclidean_distance(self,frame1, frame2):
        """
        计算两帧之间的欧几里得距离。
        """
        diff = frame1 - frame2
        # valid_mask = ~np.isnan(diff).any(axis=1)  # 只考虑非缺失的关键点 二维需要
        valid_mask = ~np.isnan(diff)  # 一维是这个
        valid_diff = diff[valid_mask]

        return np.linalg.norm(valid_diff) if len(valid_diff) > 0 else 0

    def dtw_re(self,video_A, video_B, distance_func):
        """
        使用 DTW 算法对齐两个视频的帧序列。
        """
        n, m = len(video_A), len(video_B)

        # 初始化距离矩阵和累积距离矩阵
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i][j] = distance_func(video_A[i], video_B[j])

        C = np.zeros((n, m))
        C[0][0] = D[0][0]

        # 动态规划填表
        for i in range(1, n):
            C[i][0] = C[i - 1][0] + D[i][0]
        for j in range(1, m):
            C[0][j] = C[0][j - 1] + D[0][j]
        for i in range(1, n):
            for j in range(1, m):
                C[i][j] = D[i][j] + min(C[i - 1][j], C[i][j - 1], C[i - 1][j - 1])

        # 回溯路径
        alignment = []
        i, j = n - 1, m - 1
        while i > 0 or j > 0:
            alignment.append((i, j))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = {
                    (i - 1, j): C[i - 1][j],
                    (i, j - 1): C[i][j - 1],
                    (i - 1, j - 1): C[i - 1][j - 1]
                }
                i, j = min(candidates, key=candidates.get)
        alignment.append((0, 0))
        alignment.reverse()

        return alignment, C[-1][-1]



    def dwt_keypoints(self):

        if self.video_A_raw and self.video_B_raw and self.video_A_rect and self.video_B_rect:
            processed_kps_1 = np.array(self.video_A_raw)
            processed_kps_2 = np.array(self.video_B_raw)

            # processed_kps_1 = preprocess_video(np.array(video_A_raw),np.array(video_A_rect))
            # processed_kps_2 = preprocess_video(np.array(video_B_raw),np.array(video_B_rect))

            # 提取关键点做dtw  肩膀到脚 这样提取
            selected_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            part1 = processed_kps_1[:, selected_indices, :]  # [T, N, 2]
            part2 = processed_kps_2[:, selected_indices, :]  # [T, N, 2]

            # reshape to [T, N*2]
            part1 = part1.reshape(part1.shape[0], -1)
            part2 = part2.reshape(part2.shape[0], -1)

            # alignment_path, total_distance = dtw_re(processed_kps_1, processed_kps_2, euclidean_distance)
            alignment_path, total_distance = self.dtw_re(part1, part2, self.euclidean_distance)
            print(total_distance)
            print(np.array(alignment_path).shape)
            print(alignment_path)

            # for i, j in alignment_path:
            #     # plot_frame_pair(np.array(video_A_raw)[i], np.array(video_B_raw)[j], i, j, save=False)
            #     # plot_pose_pair_blank(np.array(video_A_raw)[i], np.array(video_B_raw)[j],i,j)
            #     # 查看俩个视频
            #     plot_enlarged_pose_pair(np.array(self.video_A_raw)[i], np.array(self.video_B_raw)[j], i, j)

            self._combine_id = alignment_path

            # 保存对比的俩个pose帧
            save_pose_video(alignment_path, np.array(self.video_A_raw),np.array(self.video_B_raw),self.combine_uuid)

    def run(self) -> None:
        #启动线程

        self.dwt_keypoints()

        if(os.path.exists(self.combine_uuid)):
            self.finsh_signal.emit()



if __name__ == '__main__':
    dtwobj = DTW()
    # 拼接上全路径再给出来
    dtwobj.setparam("testvideos/020b97fb614fc01fd32dc5190610ce62_20230826111023 00_00_00-00_00_02.json",
                    "testvideos/089b923eb44a14c247a0fddea426fed8_20230826094716 00_00_00-00_00_02.json",
                    "testvideos/020b97fb614fc01fd32dc5190610ce62_20230826111023 00_00_00-00_00_02-089b923eb44a14c247a0fddea426fed8_20230826094716 00_00_00-00_00_02.mp4")
    dtwobj.readjson()
    dtwobj.dwt_keypoints()