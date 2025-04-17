# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/11 10:19
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


video_A_raw=[]
video_B_raw=[]

video_A_rect = []
video_B_rect = []

with open("testvideos/result1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

    print(data)

    for key,value in data.items():
        if len(value[-1])==4:
            video_A_raw.append(value[:-1])
            video_A_rect.append(value[-1])


    # video_A_raw = interpolate_missing_keypoints(video_A_raw)

with open("testvideos/result2.json", "r", encoding="utf-8") as f2:
    data = json.load(f2)
    print(data)
    for key, value in data.items():
        if len(value[-1])==4:
            video_B_raw.append(value[:-1])
            video_B_rect.append(value[-1])

    # video_B_raw = interpolate_missing_keypoints(video_B_raw)


def euclidean_distance(frame1, frame2):
    """
    计算两帧之间的欧几里得距离。
    """
    diff = frame1 - frame2
    valid_mask = ~np.isnan(diff).any(axis=1)  # 只考虑非缺失的关键点
    valid_diff = diff[valid_mask]
    return np.linalg.norm(valid_diff) if len(valid_diff) > 0 else 0


# ========================
# DTW 算法
# ========================

def dtw(video_A, video_B, distance_func):
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


#位置
def normalize_by_given_bbox_center(kps, centers):
    """
    用外部传入的 bbox 中心点对关键点进行归一化
    :param kps: [T, 17, 2] — 每帧17个关键点
    :param centers: [T, 4] — 每帧的 bbox，格式为 [x_min, y_min, x_max, y_max]
    :return: 归一化后关键点坐标
    """
    cx = (centers[:, 0] + centers[:, 2]) / 2  # bbox中心x
    cy = (centers[:, 1] + centers[:, 3]) / 2  # bbox中心y
    center = np.stack([cx, cy], axis=-1)[:, np.newaxis, :]  # [T, 1, 2]

    return kps - center  # 每帧关键点 - 中心点



"""
kps - center	平移归一化，消除不同位置对比干扰
/ height	尺度归一化，消除远近或人高人矮的影响
"""
# 位置+尺度
def normalize_by_given_bbox_center_and_scale(kps, centers):
    cx = (centers[:, 0] + centers[:, 2]) / 2
    cy = (centers[:, 1] + centers[:, 3]) / 2
    center = np.stack([cx, cy], axis=-1)[:, np.newaxis, :]  # [T, 1, 2]

    height = centers[:, 3] - centers[:, 1]  # y_max - y_min
    height = np.clip(height, 1e-6, None)  # 避免除0
    height = height[:, np.newaxis, np.newaxis]  # [T, 1, 1]

    return (kps - center) / height


def interpolate_and_smooth_point(series, method='linear', window=30):
    T = len(series)
    x = np.arange(T)
    valid = ~np.isnan(series)

    # 插值
    f = interp1d(x[valid], series[valid], kind=method, fill_value="extrapolate")
    interp_series = f(x)

    # 平滑
    smooth_series = pd.Series(interp_series).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


    print("原始数据长度:",T)
    print("差值的数据长度:",len(interp_series))
    print("平滑之后的数据长度:",len(smooth_series))
    return interp_series, smooth_series


def plot_compare_keypoint_trajectory(keypoints, kpt_index=0):
    """
    显示某个关键点在所有帧中x/y坐标的插值和平滑效果对比
    """

    x_series = keypoints[:, kpt_index, 0]
    y_series = keypoints[:, kpt_index, 1]

    x_interp, x_smooth = interpolate_and_smooth_point(x_series,15)
    y_interp, y_smooth = interpolate_and_smooth_point(y_series,15)

    frames = np.arange(len(x_series))

    plt.figure(figsize=(12, 5))

    # x轨迹
    plt.subplot(1, 2, 1)
    plt.plot(frames, x_series, 'ro-', label='srcx', alpha=0.5)
    plt.plot(frames, x_interp, 'b--', label='insertx')
    plt.plot(frames, x_smooth, 'g-', label='smoothx')
    plt.title(f'keypoint{kpt_index} X')
    plt.xlabel("frame")
    plt.ylabel("X")
    plt.legend()

    # y轨迹
    plt.subplot(1, 2, 2)
    plt.plot(frames, y_series, 'ro-', label='srcy', alpha=0.5)
    plt.plot(frames, y_interp, 'b--', label='inserty')
    plt.plot(frames, y_smooth, 'g-', label='smoothy')
    plt.title(f'keypoint{kpt_index} Y')
    plt.xlabel("frame")
    plt.ylabel("Y")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_compare_keypoint_trajectory2(keypoints,kpt_index=0):

    T,J,D = keypoints.shape
    x_series = keypoints[:, kpt_index, 0]
    y_series = keypoints[:, kpt_index, 1]

    x_interp, x_smooth = interpolate_and_smooth_point(x_series, 15)
    y_interp, y_smooth = interpolate_and_smooth_point(y_series, 15)

    kps_flatten = normalize_by_given_bbox_center(x_smooth, np.array(video_B_rect))

    return kps_flatten


def plot_alignment_path(alignment_path, C):
    """
    绘制 DTW 对齐路径和累积距离矩阵。
    """
    plt.figure(figsize=(12, 6))

    # 绘制累积距离矩阵
    plt.subplot(1, 2, 1)
    plt.imshow(C, cmap='hot', origin='lower')
    plt.title("Cumulative Distance Matrix")
    plt.colorbar()

    # 绘制对齐路径
    plt.subplot(1, 2, 2)
    alignment_x, alignment_y = zip(*alignment_path)
    plt.plot(alignment_y, alignment_x, label="Alignment Path", color="red")
    plt.title("DTW Alignment Path")
    plt.xlabel("Video B Frames")
    plt.ylabel("Video A Frames")
    plt.legend()

    plt.tight_layout()
    plt.show()


def dwt_keypoints():
    processed_kps_1 = plot_compare_keypoint_trajectory2(np.array(video_A_raw),kpt_index=9)
    processed_kps_2 = plot_compare_keypoint_trajectory2(np.array(video_B_raw), kpt_index=9)

    alignment_path, total_distance = dtw(processed_kps_1, processed_kps_2, euclidean_distance)

    # 输出结果
    print("Alignment Path:", alignment_path)
    print("Total Distance:", total_distance)

    C = np.zeros((len(processed_kps_1), len(processed_kps_2)))
    for i in range(len(processed_kps_1)):
        for j in range(len(processed_kps_2)):
            C[i][j] = euclidean_distance(processed_kps_1[i], processed_kps_2[j])

    plot_alignment_path(alignment_path, C)


dwt_keypoints()

# print("---:",video_B_raw)
# print(np.array(video_B_raw).shape)
# print(np.array(video_B_rect).shape)

# plot_compare_keypoint_trajectory(np.array(video_B_raw), kpt_index=9)