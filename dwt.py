# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/10 14:30
import json

import numpy as np
import matplotlib.pyplot as plt


# ========================
# 数据预处理
# ========================

def forward_fill_keypoints(video_frames):
    """
    向前填充缺失的关键点。
    """
    filled_frames = []
    last_valid_frame = None
    for frame in video_frames:
        if np.all(np.isnan(frame)):  # 如果整个帧都是 NaN（完全缺失）
            if last_valid_frame is not None:
                frame = last_valid_frame  # 用上一帧填补
        else:
            last_valid_frame = frame  # 更新有效帧
        filled_frames.append(frame)
    return np.array(filled_frames)


def normalize_keypoints(video_frames):
    """
    归一化关键点坐标到 [0, 1] 范围。
    """
    normalized_frames = []
    for frame in video_frames:
        max_val = np.nanmax(np.abs(frame))  # 忽略 NaN 计算最大值
        if max_val > 0:
            frame = frame / max_val
        normalized_frames.append(frame)
    return np.array(normalized_frames)


def preprocess_video(video_frames):
    """
    预处理视频帧：向前填充 + 归一化。
    """
    video_frames = forward_fill_keypoints(video_frames)  # 填补缺失帧
    video_frames = normalize_keypoints(video_frames)  # 归一化坐标
    return video_frames


# ========================
# 距离度量
# ========================

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


# ========================
# 可视化
# ========================

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


# ========================
# 每个关键点的对比图表显示
# ========================

def plot_keypoint_comparison_one_by_one(video_A, video_B, alignment_path):
    """
    逐个显示对齐帧中每个关键点的对比图表。
    """
    num_keypoints = video_A.shape[1]  # 关键点数量（假设为 17）

    # 提取对齐路径中的帧
    aligned_frames_A = [video_A[i] for i, _ in alignment_path]
    aligned_frames_B = [video_B[j] for _, j in alignment_path]

    for k in range(num_keypoints):
        # 获取当前关键点的 x 和 y 坐标
        keypoint_A_x = [frame[k][0] for frame in aligned_frames_A]
        keypoint_A_y = [frame[k][1] for frame in aligned_frames_A]
        keypoint_B_x = [frame[k][0] for frame in aligned_frames_B]
        keypoint_B_y = [frame[k][1] for frame in aligned_frames_B]

        # 创建图表
        plt.figure(figsize=(12, 6))

        # 绘制 x 坐标对比
        plt.subplot(1, 2, 1)
        plt.plot(keypoint_A_x, label="Video A (x)", color="blue")
        plt.plot(keypoint_B_x, label="Video B (x)", color="orange")
        plt.title(f"Keypoint {k + 1} - X Coordinate")
        plt.legend()
        plt.xlabel("Frame Index")
        plt.ylabel("X Value")

        # 绘制 y 坐标对比
        plt.subplot(1, 2, 2)
        plt.plot(keypoint_A_y, label="Video A (y)", color="blue")
        plt.plot(keypoint_B_y, label="Video B (y)", color="orange")
        plt.title(f"Keypoint {k + 1} - Y Coordinate")
        plt.legend()
        plt.xlabel("Frame Index")
        plt.ylabel("Y Value")

        plt.tight_layout()

        # 显示当前关键点的图表
        print(f"Displaying Keypoint {k + 1}...")
        plt.show()  # 暂停显示，用户关闭窗口后继续

        # 清空图表以准备下一次绘制
        plt.clf()


def plot_keypoint_comparison(video_A, video_B, alignment_path):
    """
    绘制对齐帧中每个关键点的对比图表。
    """
    num_keypoints = video_A.shape[1]  # 关键点数量（假设为 17）

    # 提取对齐路径中的帧
    aligned_frames_A = [video_A[i] for i, _ in alignment_path]
    aligned_frames_B = [video_B[j] for _, j in alignment_path]

    # 创建子图
    fig, axes = plt.subplots(num_keypoints, 2, figsize=(15, 4 * num_keypoints))
    fig.suptitle("Keypoint Comparison Between Video A and Video B", fontsize=20)

    for k in range(num_keypoints):
        # 获取当前关键点的 x 和 y 坐标
        keypoint_A_x = [frame[k][0] for frame in aligned_frames_A]
        keypoint_A_y = [frame[k][1] for frame in aligned_frames_A]
        keypoint_B_x = [frame[k][0] for frame in aligned_frames_B]
        keypoint_B_y = [frame[k][1] for frame in aligned_frames_B]

        # 计算误差
        error_x = np.abs(np.array(keypoint_A_x) - np.array(keypoint_B_x))
        error_y = np.abs(np.array(keypoint_A_y) - np.array(keypoint_B_y))

        # 绘制 x 坐标对比
        axes[k, 0].plot(keypoint_A_x, label="Video A (x)", color="blue")
        axes[k, 0].plot(keypoint_B_x, label="Video B (x)", color="orange")
        axes[k, 0].set_title(f"Keypoint {k + 1} - X Coordinate")
        axes[k, 0].legend()
        axes[k, 0].set_xlabel("Frame Index")
        axes[k, 0].set_ylabel("X Value")

        # 绘制 y 坐标对比
        axes[k, 1].plot(keypoint_A_y, label="Video A (y)", color="blue")
        axes[k, 1].plot(keypoint_B_y, label="Video B (y)", color="orange")
        axes[k, 1].set_title(f"Keypoint {k + 1} - Y Coordinate")
        axes[k, 1].legend()
        axes[k, 1].set_xlabel("Frame Index")
        axes[k, 1].set_ylabel("Y Value")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_keypoint_errors(video_A, video_B, alignment_path):
    """
    绘制每个关键点的误差分布图。
    """
    num_keypoints = video_A.shape[1]  # 关键点数量（假设为 17）

    # 提取对齐路径中的帧
    aligned_frames_A = [video_A[i] for i, _ in alignment_path]
    aligned_frames_B = [video_B[j] for _, j in alignment_path]

    # 计算每个关键点的平均误差
    errors = []
    for k in range(num_keypoints):
        keypoint_A = np.array([frame[k] for frame in aligned_frames_A])
        keypoint_B = np.array([frame[k] for frame in aligned_frames_B])
        diff = keypoint_A - keypoint_B
        valid_mask = ~np.isnan(diff).any(axis=1)  # 忽略缺失值
        valid_diff = diff[valid_mask]
        error = np.mean(np.linalg.norm(valid_diff, axis=1)) if len(valid_diff) > 0 else 0
        errors.append(error)

    # 绘制误差分布图
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_keypoints + 1), errors, color="skyblue")
    plt.title("Average Euclidean Error per Keypoint")
    plt.xlabel("Keypoint Index")
    plt.ylabel("Average Error")
    plt.xticks(range(1, num_keypoints + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def interpolate_missing_keypoints(keypoints):
    keypoints = np.array(keypoints, dtype=np.float32)  # shape: (frames, num_kpts, 2)
    frames, num_kpts, _ = keypoints.shape

    for kp_idx in range(num_kpts):
        for dim in range(2):  # x 和 y 分别插值
            values = keypoints[:, kp_idx, dim]
            mask = (values != 0)  # 非0表示检测到

            if np.sum(mask) < 2:
                continue  # 数据太少不插值

            # 找到非零的帧
            valid_idx = np.where(mask)[0]
            valid_values = values[mask]

            # 插值
            interpolated = np.interp(np.arange(frames), valid_idx, valid_values)
            # 替换 0 的地方
            keypoints[:, kp_idx, dim][~mask] = interpolated[~mask]

    return keypoints




# ========================
# 示例数据
# ========================

# 模拟两个视频帧序列，每帧包含 17 个关键点（x, y 坐标）
# 缺失的关键点用 NaN 表示
# video_A_raw = [
#     np.full((17, 2), np.nan),  # 第 1 帧缺失
#     np.full((17, 2), np.nan),  # 第 2 帧缺失
#     np.random.rand(17, 2),  # 第 3 帧有效
#     np.random.rand(17, 2),  # 第 4 帧有效
# ]
#
# video_B_raw = [
#     np.random.rand(17, 2),  # 第 1 帧有效
#     np.random.rand(17, 2),  # 第 2 帧有效
#     np.random.rand(17, 2),  # 第 3 帧有效
# ]

video_A_raw=[]

video_B_raw=[]

with open("testvideos/result1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

    print(data)

    for key,value in data.items():
        video_A_raw.append(value)

    video_A_raw = interpolate_missing_keypoints(video_A_raw)

with open("testvideos/result2.json", "r", encoding="utf-8") as f2:
    data = json.load(f2)
    for key, value in data.items():
        video_B_raw.append(value)

    video_B_raw = interpolate_missing_keypoints(video_B_raw)


print(video_A_raw)
print(video_B_raw)


if 1:
    # ========================
    # 主程序
    # ========================

    # 预处理视频帧
    video_A_processed = preprocess_video(video_A_raw)
    video_B_processed = preprocess_video(video_B_raw)

    # 应用 DTW 算法
    alignment_path, total_distance = dtw(video_A_processed, video_B_processed, euclidean_distance)

    # 输出结果
    print("Alignment Path:", alignment_path)
    print("Total Distance:", total_distance)

    # 可视化对齐结果
    C = np.zeros((len(video_A_processed), len(video_B_processed)))
    for i in range(len(video_A_processed)):
        for j in range(len(video_B_processed)):
            C[i][j] = euclidean_distance(video_A_processed[i], video_B_processed[j])

    plot_alignment_path(alignment_path, C)

    # ========================
    # 主程序扩展
    # ========================

    # 调用函数逐个显示关键点对比图表
    plot_keypoint_comparison_one_by_one(video_A_processed, video_B_processed, alignment_path)

    # 调用函数绘制关键点对比图表
    # plot_keypoint_comparison(video_A_processed, video_B_processed, alignment_path)

    # 调用函数绘制关键点误差分布图
    # plot_keypoint_errors(video_A_processed, video_B_processed, alignment_path)