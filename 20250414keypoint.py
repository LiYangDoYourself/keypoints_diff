# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/14 10:31
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtw import dtw
from sklearn.metrics.pairwise import cosine_similarity
from calculate_angle import *


video_A_raw=[]
video_B_raw=[]

video_A_rect = []
video_B_rect = []

with open("testvideos/020b97fb614fc01fd32dc5190610ce62_20230826111023 00_00_00-00_00_02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

    print(data)

    for key,value in data.items():
        if len(value[-1])==4:
            video_A_raw.append(value[:-1])
            video_A_rect.append(value[-1])


    # video_A_raw = interpolate_missing_keypoints(video_A_raw)

with open("testvideos/089b923eb44a14c247a0fddea426fed8_20230826094716 00_00_00-00_00_02.json", "r", encoding="utf-8") as f2:
    data = json.load(f2)
    print(data)
    for key, value in data.items():
        if len(value[-1])==4:
            video_B_raw.append(value[:-1])
            video_B_rect.append(value[-1])





def euclidean_distance(frame1, frame2):
    """
    计算两帧之间的欧几里得距离。
    """
    diff = frame1 - frame2
    # valid_mask = ~np.isnan(diff).any(axis=1)  # 只考虑非缺失的关键点 二维需要
    valid_mask = ~np.isnan(diff)    # 一维是这个
    valid_diff = diff[valid_mask]

    return np.linalg.norm(valid_diff) if len(valid_diff) > 0 else 0


# ========================
# DTW 算法
# ========================
def dtw_re(video_A, video_B, distance_func):
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


#位置归一化
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

# 返回插入值和平滑之后的值
def interpolate_and_smooth_point(series, method='linear', window=30):
    T = len(series)
    x = np.arange(T)
    valid = ~np.isnan(series)

    if np.sum(valid) < 2:
        # 如果有效点太少，就全部设为0
        interp_series = np.zeros_like(series)
    else:
        # 插值
        f = interp1d(x[valid], series[valid], kind=method, fill_value="extrapolate")
        interp_series = f(x)

    # 平滑
    smooth_series = pd.Series(interp_series).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return interp_series, smooth_series

# 平滑关键点
def smooth_keypoints(kps, method='linear', window=30):
    T, J, D = kps.shape
    interp_kps = np.zeros_like(kps)
    smooth_kps = np.zeros_like(kps)

    for j in range(J):
        for d in range(D):
            series = kps[:, j, d]
            interp, smooth = interpolate_and_smooth_point(series, method=method, window=window)
            interp_kps[:, j, d] = interp
            smooth_kps[:, j, d] = smooth

    # return interp_kps, smooth_kps
    return smooth_kps

# 预处理视频帧：向前填充 + 归一化。
def preprocess_video(video_frames,video_centers):
    """
    预处理视频帧：向前填充 + 归一化。
    """
    video_frames_keypoint = smooth_keypoints(video_frames)  # 填补缺失帧, 平滑关键点
    norm_keypoint = normalize_by_given_bbox_center(video_frames_keypoint,video_centers)   # 归一化坐标

    return norm_keypoint

def plot_dtw_alignment(alignment, title='DTW 匹配路径'):
    """
    可视化 DTW 匹配路径
    """
    path = alignment.index1, alignment.index2
    plt.figure(figsize=(8, 6))
    plt.plot(path[0], path[1], lw=1)
    plt.xlabel('序列1帧数')
    plt.ylabel('序列2帧数')
    plt.title(title)
    plt.grid(True)
    plt.show()

# def plot_skeleton(kps, ax, color='r'):
#     """
#     绘制单帧 17 个关键点
#     kps: [17, 2]
#     """
#     ax.scatter(kps[:, 0], -kps[:, 1], c=color, s=20, label='Skeleton', alpha=0.6)
#
#     # 常见人体骨架连接顺序 (COCO格式部分)
#     skeleton = [
#         (0, 1), (1, 2), (2, 3), (3, 4),
#         (1, 5), (5, 6), (6, 7),
#         (1, 8), (8, 9), (9, 10),
#         (8, 12), (12, 13), (13, 14),
#         (0, 15), (0, 16)
#     ]
#     for (i, j) in skeleton:
#         if not np.any(np.isnan(kps[[i, j]])):
#             ax.plot(kps[[i, j], 0], -kps[[i, j], 1], color=color, alpha=0.5)
#
# def plot_matched_skeleton_pairs(norm1, norm2, alignment, interval=10, max_pairs=10):
#     """
#     绘制通过 DTW 匹配的帧对的骨架对比图
#     """
#     indices1 = alignment.index1
#     indices2 = alignment.index2
#
#     num_plots = min(len(indices1), max_pairs)
#     for idx in range(0, num_plots, interval):
#         i = indices1[idx]
#         j = indices2[idx]
#
#         fig, ax = plt.subplots(figsize=(4, 4))
#         ax.set_title(f"match id: norm1[{i}] vs norm2[{j}]")
#         plot_skeleton(norm1[i], ax, color='red')
#         plot_skeleton(norm2[j], ax, color='blue')
#         ax.legend(['norm1', 'norm2'])
#         ax.axis('equal')
#         ax.invert_yaxis()
#         plt.show()


# COCO骨架连接顺序
# COCO_PAIRS = [
#     [0, 1], [1, 3], [0, 2], [2, 4],
#     [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
#     [5, 11], [6, 12], [11, 12],
#     [11, 13], [13, 15], [12, 14], [14, 16]
# ]
COCO_PAIRS = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]


# 示例角度定义：每个角度由三个关键点编号构成（中间点是角点）
# 示例：肘部角度（肩 - 肘 - 手腕）
ANGLE_JOINTS = [
    (5, 7, 9),   # 左臂：肩-肘-腕
    (6, 8, 10),   # 右臂：肩-肘-腕
    (11,13,15),# 左腿：髋-膝-踝
    (12,14,16),# 右腿：髋-膝-踝
]

def calculate_angle(a, b, c):
    """计算角度，b为顶点"""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)



def plot_skeleton(kps, ax, color='r', label=None):

    ax.scatter(kps[:, 0], -kps[:, 1], c=color, s=20, alpha=0.7, label=label)
    for p1, p2 in COCO_PAIRS:
        if not (np.any(np.isnan(kps[p1])) or np.any(np.isnan(kps[p2]))):
            ax.plot([kps[p1, 0], kps[p2, 0]],
                    [-kps[p1, 1], -kps[p2, 1]],
                    color=color, linewidth=1.5, alpha=0.6)

def is_valid_point(p):
    return not np.allclose(p, 0)

def safe_calculate_angle(a, b, c):
    if not (is_valid_point(a) and is_valid_point(b) and is_valid_point(c)):
        return None
    return calculate_angle(a, b, c)



def plot_frame_pair(kps1, kps2, i, j, save=False, outdir='frames'):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_skeleton(kps1, ax, color='red', label='Video1')
    plot_skeleton(kps2, ax, color='blue', label='Video2')

    # 锁定坐标轴范围（根据你的图像分辨率调整）
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 928)

    # ax.text(348, 687, "Hello", color='green')
    # ax.plot(348, 687, 'o', color='green', markersize=10)

    offset = 10  # 可以调大试试
    # 显示角度
    for (a, b, c) in ANGLE_JOINTS:
        if max(a, b, c) < len(kps1):
            angle1 = safe_calculate_angle(kps1[a], kps1[b], kps1[c])
            if angle1 is not None and is_valid_point(kps1[b, 1]):
                # print(kps1[b, 0], kps1[b, 1]-offset)
                # ax.text(kps1[b, 0], kps1[b, 1]-offset, f'{angle1:.1f}°', color='green')
                ax.text(kps1[b, 0], kps1[b, 1]-offset, f'{angle1:.1f}°', color='red', fontsize=8,zorder=20,alpha=1.0)

        if max(a, b, c) < len(kps2):
            angle2 = safe_calculate_angle(kps2[a], kps2[b], kps2[c])
            if angle2 is not None and is_valid_point(kps2[b, 1]):
                ax.text(kps2[b, 0], kps2[b, 1]-offset, f'{angle2:.1f}°', color='blue', fontsize=8,zorder=20,alpha=1.0)

    ax.set_title(f'Frame Match: Video1[{i}] vs Video2[{j}]')
    ax.axis('equal')
    ax.axis('off')
    ax.legend()

    if save:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/match_{i:03d}_{j:03d}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()



# def plot_skeleton(kps, ax, color='r', label=None):
#     ax.scatter(kps[:, 0], -kps[:, 1], c=color, s=20, alpha=0.7, label=label)
#     for p1, p2 in COCO_PAIRS:
#         if not (np.any(np.isnan(kps[p1])) or np.any(np.isnan(kps[p2]))):
#             ax.plot([kps[p1, 0], kps[p2, 0]],
#                     [-kps[p1, 1], -kps[p2, 1]],
#                     color=color, linewidth=1.5, alpha=0.6)
#
# def plot_frame_pair(kps1, kps2, i, j, save=False, outdir='frames'):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     plot_skeleton(kps1, ax, color='red', label='Video1')
#     plot_skeleton(kps2, ax, color='blue', label='Video2')
#     ax.set_title(f'Frame Match: Video1[{i}] vs Video2[{j}]')
#     ax.axis('equal')
#     ax.axis('off')
#     ax.legend()
#     if save:
#         import os
#         os.makedirs(outdir, exist_ok=True)
#         plt.savefig(f'{outdir}/match_{i:03d}_{j:03d}.png', bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()



def dwt_keypoints():

    # 这个是平滑+归一化
    # processed_kps_1 = preprocess_video(np.array(video_A_raw),np.array(video_A_rect))
    # processed_kps_2 = preprocess_video(np.array(video_B_raw),np.array(video_B_rect))
    # alignment_path, total_distance = dtw_re(processed_kps_1, processed_kps_2, euclidean_distance)

    # 这个是原始数据 [T,17,2]
    # processed_kps_1 = np.array(video_A_raw)
    # processed_kps_2 = np.array(video_B_raw)
    # alignment_path, total_distance = dtw_re(processed_kps_1,processed_kps_2,euclidean_distance)

    processed_kps_1 = np.array(video_A_raw).reshape(np.array(video_A_raw).shape[0], -1)
    processed_kps_2 = np.array(video_B_raw).reshape(np.array(video_B_raw).shape[0], -1)
    # distance_func = lambda x, y: np.linalg.norm(x - y)

    alignment = dtw(processed_kps_1, processed_kps_2)  #dist_method=distance_func

    # plot_dtw_alignment(alignment)

    # plot_matched_skeleton_pairs(np.array(video_A_raw), np.array(video_B_raw), alignment, interval=10, max_pairs=10)

    # 获取匹配路径（index 对应）
    print(alignment.index1)
    print(alignment.index2)
    path = list(zip(alignment.index1, alignment.index2))  #
    print(f"DTW distance: {alignment.distance}")
    for i, j in path:
        plot_frame_pair(np.array(video_A_raw)[i], np.array(video_B_raw)[j], i, j, save=False)

        # plot_pose_pair_blank(kps1, kps2)

    # 输出结果
    # print("Alignment Path:", alignment_path)
    # print("Total Distance:", total_distance)

def compute_cosine_similarity(kps1, kps2):
    """
    计算余弦相似度
    kps1, kps2: [T, 17, 2]
    """
    # 展开成 [T, 34]
    kps1_flat = kps1.reshape(kps1.shape[0], -1)
    kps2_flat = kps2.reshape(kps2.shape[0], -1)

    print("kps1_flat.shape:",kps1_flat.shape)
    print("kps2_flat.shape:",kps2_flat.shape)

    # 归一化每个向量
    kps1_norm = kps1_flat / np.linalg.norm(kps1_flat, axis=1, keepdims=True)
    kps2_norm = kps2_flat / np.linalg.norm(kps2_flat, axis=1, keepdims=True)

    # 计算余弦相似度
    cos_sim = cosine_similarity(kps1_norm, kps2_norm)
    return cos_sim


def extract_body_parts(kps, part='upper_body'):
    """
    根据部位提取关键点
    part: 'upper_body' 或 'lower_body'
    kps: [T, 17, 2]
    返回：提取出的部位的关键点数据
    """
    # 假设关键点编号为：
    # 头部：0
    # 肩膀：1, 2
    # 手臂：3-6
    # 上半身：[0, 1, 2, 3, 4, 5, 6]
    # 下半身：[7-16]

    if part == 'upper_body':
        # 上半身关键点：头部 + 肩膀 + 手臂
        indices = [0, 1, 2, 3, 4, 5, 6]
    elif part == 'lower_body':
        # 下半身关键点：臀部 + 大腿 + 小腿
        indices = list(range(7, 17))
    elif part== 'point':
        indices = [5,6,7,8,9,10,11,12,13,14,15,16]
    else:
        indices = int(part)

    return kps[:, indices, :]


def compare_body_parts(kps1, kps2, part='upper_body'):
    """
    比较不同部位的余弦相似度
    """
    # 提取部位关键点
    kps1_part = extract_body_parts(kps1, part)
    kps2_part = extract_body_parts(kps2, part)


    print("kps1_part shape",kps1_part.shape)
    print("kps2_part shape",kps2_part.shape)

    # 计算余弦相似度
    cos_sim = compute_cosine_similarity(kps1_part, kps2_part)

    return cos_sim


def extract_aligned_keypoints(kps1, kps2, path):
    """
    根据 DTW 对齐路径提取对齐的关键点数据
    返回：两个新的关键点数组，对齐后的
    """
    aligned_kps1 = []
    aligned_kps2 = []

    for i, j in path:
        aligned_kps1.append(kps1[i])
        aligned_kps2.append(kps2[j])

    # 转为 ndarray: [len(path), 17, 2]
    aligned_kps1 = np.array(aligned_kps1)
    aligned_kps2 = np.array(aligned_kps2)

    return aligned_kps1, aligned_kps2





# 完整的获取视频流 并保存成帧 然后优化 并获匹配的关键帧数据 ,然后画出关键帧的匹配图
def dwt_keypoints2():

    processed_kps_1 = np.array(video_A_raw)
    processed_kps_2 = np.array(video_B_raw)

    # processed_kps_1 = preprocess_video(np.array(video_A_raw),np.array(video_A_rect))
    # processed_kps_2 = preprocess_video(np.array(video_B_raw),np.array(video_B_rect))

    # 提取关键点做dtw  肩膀到脚 这样提取
    selected_indices = [5,6,7,8,9,10,11,12,13,14,15,16]
    part1 = processed_kps_1[:, selected_indices, :]  # [T, N, 2]
    part2 = processed_kps_2[:, selected_indices, :]  # [T, N, 2]

    print(part1.shape)
    print(part2.shape)

    # reshape to [T, N*2]
    part1 = part1.reshape(part1.shape[0], -1)
    part2 = part2.reshape(part2.shape[0], -1)

    # alignment_path, total_distance = dtw_re(processed_kps_1, processed_kps_2, euclidean_distance)
    alignment_path, total_distance = dtw_re(part1, part2, euclidean_distance)
    print(total_distance)
    print(np.array(alignment_path).shape)
    print(alignment_path)

    for i, j in alignment_path:
        # plot_frame_pair(np.array(video_A_raw)[i], np.array(video_B_raw)[j], i, j, save=False)
        # plot_pose_pair_blank(np.array(video_A_raw)[i], np.array(video_B_raw)[j],i,j)
        plot_enlarged_pose_pair(np.array(video_A_raw)[i], np.array(video_B_raw)[j],i,j)

    # 保存对比的俩个pose帧
    # save_pose_video(alignment_path, np.array(video_A_raw),np.array(video_B_raw))

    """
    相似帧之后计算关键点角度
    1 计算垂直角度
    """


    # aligned_kps1, aligned_kps2 = extract_aligned_keypoints(np.array(video_A_raw), np.array(video_B_raw), alignment_path)

    # print("aligned_kps1 shape",aligned_kps1.shape)
    # print("aligned_kps2 shape",aligned_kps2.shape)

    # cos_sim = compare_body_parts(aligned_kps1, aligned_kps2, part='10')
    # print("cos_sim:",cos_sim)
    # print("cos_sim shape",cos_sim.shape)

dwt_keypoints2()



# 每对匹配帧的骨架对比图；
# 动作相似度评分；
# 动作关键帧时间轴图；