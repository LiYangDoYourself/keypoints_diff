# -*- coding: utf-8 -*-
# author: liyang
# time: 2025/4/15 11:23
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
# 示例骨架连接定义（你可以换成你自己项目的）

videowh = (1280,1080)  #(1280,928)

SKELETON = [
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

def safe_calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_pose(ax, kps, skeleton, angle_joints, color='red', label_prefix=''):
    # 骨架线
    for i, j in skeleton:
        if i < len(kps) and j < len(kps):
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=2)

    # 关键点
    ax.scatter(kps[:, 0], kps[:, 1], c=color, s=20, label=label_prefix)

    # 角度标注
    for (a, b, c) in angle_joints:
        if max(a, b, c) < len(kps):
            angle = safe_calculate_angle(kps[a], kps[b], kps[c])
            if angle is not None:
                ax.text(kps[b, 0], kps[b, 1] - 5, f'{angle:.1f}°', color=color, fontsize=8)


def normalize_and_scale_kps(kps, scale=2.5, center_to=(640, 464)):
    """
    将关键点平移至中心并放大。
    :param kps: 原始关键点 (17, 2)
    :param scale: 放大倍数
    :param center_to: 放大后中心对准的位置
    :return: 新的关键点 (17, 2)
    """
    center = np.mean(kps, axis=0)
    kps_centered = kps - center  # 平移到原点
    kps_scaled = kps_centered * scale
    kps_translated = kps_scaled + np.array(center_to)
    return kps_translated

def plot_enlarged_pose_pair(kps1, kps2,i,j, scale=2.5, save=False,return_image=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # 处理关键点
    kps1_show = normalize_and_scale_kps(kps1, scale=scale)
    kps2_show = normalize_and_scale_kps(kps2, scale=scale)

    for ax, kps, color, title in zip([ax1, ax2], [kps1_show, kps2_show], ['red', 'blue'], [f'Video1_{i}', f'Video2_{j}']):
        draw_pose(ax, kps, SKELETON, ANGLE_JOINTS, color=color)
        ax.set_xlim(0, videowh[0])   #1280
        ax.set_ylim(videowh[1], 0)   #928
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    if return_image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image  # 返回RGB图像

    if save:
        plt.savefig('enlarged_pose_pair.png', dpi=200)
        plt.close()
    else:
        plt.show()


def save_pose_video(alignment_path, kps1_list,kps2_list,output_path='testvideos/pose_video.mp4', fps=25):
    # 先画第一帧确定图像大小

    # h, w, _ = 1920,1080,1

    h,w = videowh[1],videowh[0]
    for i, j in alignment_path:
        frame = plot_enlarged_pose_pair(kps1_list[i], kps2_list[j], 0, 0, return_image=True)
        h, w, _ = frame.shape
        break


    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i, j in alignment_path:
        frame = plot_enlarged_pose_pair(kps1_list[i], kps2_list[j], i, j, return_image=True)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"✅ 视频已保存至：{output_path}")

    return output_path

def plot_pose_pair_blank(kps1, kps2,i,j,angle_joints=ANGLE_JOINTS, skeleton=SKELETON, save=False, return_image=False,out_path='pose_pair.png'):
    # 创建一个图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 25))

    # 设置左图（Video1）
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(928, 0)  # Adjust Y-axis as needed
    ax1.set_aspect('equal')
    ax1.axis('off')
    draw_pose(ax1, kps1, skeleton, angle_joints, color='red', label_prefix='Video1')

    # 设置右图（Video2）
    ax2.set_xlim(0, 1280)
    ax2.set_ylim(928, 0)  # Adjust Y-axis as needed
    ax2.set_aspect('equal')
    ax2.axis('off')
    draw_pose(ax2, kps2, skeleton, angle_joints, color='blue', label_prefix='Video2')

    # 可选：为图像添加标题

    ax1.set_title(f'Frame Match: Video1[{i}]')
    ax2.set_title(f'Frame Match: Video2[{j}]')

    if return_image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image  # 返回RGB图像

    if save:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()