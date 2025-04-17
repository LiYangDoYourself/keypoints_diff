# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/10 16:36
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.spatial.distance import cosine


# ========= 补全关键点中的 (0,0) =========
def interpolate_keypoints_dict(kpts_dict):
    sorted_frames = sorted(kpts_dict.keys())
    keypoints_list = [kpts_dict[f] for f in sorted_frames]
    keypoints_array = np.array(keypoints_list, dtype=np.float32)  # [frames, 17, 2]

    num_frames, num_kpts, _ = keypoints_array.shape

    for kp_idx in range(num_kpts):
        # 获取该关键点在所有帧中的 (x, y)
        x_vals = keypoints_array[:, kp_idx, 0]
        y_vals = keypoints_array[:, kp_idx, 1]

        # 判断缺失点：只有 (0,0) 同时成立才算是缺失
        valid_mask = ~((x_vals == 0) & (y_vals == 0))

        # 如果有效帧数小于2，无法插值，跳过
        if np.sum(valid_mask) < 2:
            continue

        valid_indices = np.where(valid_mask)[0]

        # 插值 x 和 y
        x_interp = np.interp(np.arange(num_frames), valid_indices, x_vals[valid_mask])
        y_interp = np.interp(np.arange(num_frames), valid_indices, y_vals[valid_mask])

        # 替换缺失的值
        missing_indices = np.where(~valid_mask)[0]
        keypoints_array[missing_indices, kp_idx, 0] = x_interp[missing_indices]
        keypoints_array[missing_indices, kp_idx, 1] = y_interp[missing_indices]

    # 返回新的补全后字典
    return {
        frame_id: keypoints_array[i].tolist()
        for i, frame_id in enumerate(sorted_frames)
    }



# ========= 归一化关键点，以人体边框中心为中心点 =========
def normalize_keypoints(kpts_dict):
    normed_dict = {}
    for frame_id, kpts in kpts_dict.items():
        kpts = np.array(kpts)
        valid = kpts[(kpts != 0).all(axis=1)]

        if len(valid) == 0:
            normed_dict[frame_id] = kpts.tolist()
            continue

        center = np.mean(valid, axis=0)  # 边框中心
        scale = np.max(np.linalg.norm(valid - center, axis=1)) + 1e-6

        kpts_normed = (kpts - center) / scale
        normed_dict[frame_id] = kpts_normed.tolist()

    return normed_dict


# ========= 将关键点序列拉平成1维向量用于 DTW =========
def flatten_keypoints(kpts_dict):
    sorted_keys = sorted(kpts_dict.keys())
    return np.array([
        np.array(kpts_dict[f]).flatten() for f in sorted_keys
    ])


# ========= 使用DTW进行序列对齐 =========
def run_dtw(kpts1, kpts2):
    seq1 = flatten_keypoints(kpts1)
    seq2 = flatten_keypoints(kpts2)

    dist_fun = lambda x, y: cosine(x, y)
    alignment = dtw(seq1, seq2, dist=dist_fun,keep_internals=True)

    return alignment.index1, alignment.index2, alignment.distance


# ========= 可视化关键帧对比 =========
def plot_keypoints_compare(kpts1, kpts2, frame_pairs, title="Compare"):
    for idx1, idx2 in frame_pairs:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        k1 = np.array(list(kpts1.values())[idx1])
        k2 = np.array(list(kpts2.values())[idx2])

        axs[0].scatter(k1[:, 0], -k1[:, 1], c='blue')
        axs[0].set_title(f"{title} A - Frame {idx1}")
        axs[0].axis('equal')

        axs[1].scatter(k2[:, 0], -k2[:, 1], c='green')
        axs[1].set_title(f"{title} B - Frame {idx2}")
        axs[1].axis('equal')

        plt.tight_layout()
        plt.show()


# ========= 主程序入口 =========
def main():
    import json

    # 假设你已经提取好关键点字典格式，并保存在 JSON 文件中
    with open("testvideos/result1.json", "r") as f:
        video_a_kpts = json.load(f)
    with open("testvideos/result2.json", "r") as f:
        video_b_kpts = json.load(f)

    # 保证字典键是整数
    video_a_kpts = {int(k): v for k, v in video_a_kpts.items()}
    video_b_kpts = {int(k): v for k, v in video_b_kpts.items()}

    print(video_a_kpts)
    print(video_b_kpts)

    # 1. 插值补全缺失点
    video_a_filled = interpolate_keypoints_dict(video_a_kpts)
    video_b_filled = interpolate_keypoints_dict(video_b_kpts)

    # 2. 归一化关键点
    video_a_norm = normalize_keypoints(video_a_filled)
    video_b_norm = normalize_keypoints(video_b_filled)

    # 3. DTW 匹配帧序列
    idx1_list, idx2_list, dtw_distance = run_dtw(video_a_norm, video_b_norm)

    print("🌀 DTW 总距离:", dtw_distance)
    print("✅ 匹配帧序列长度:", len(idx1_list))
    print("🔁 映射前5帧:", list(zip(idx1_list, idx2_list))[:5])

    # 4. 可视化每隔几帧的一对
    step = max(1, len(idx1_list) // 5)
    frame_pairs = list(zip(idx1_list, idx2_list))[::step]
    plot_keypoints_compare(video_a_norm, video_b_norm, frame_pairs, title="KeyFrame Compare")


if __name__ == "__main__":
    main()
