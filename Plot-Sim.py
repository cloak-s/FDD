# # import torch
# #
# # def pearson_corr(x, y, eps=1e-8):
# #     """
# #     x, y: torch.Tensor, shape [N] 或 [B, N]
# #     如果是 [B, N]，则对最后一维计算 Pearson
# #     """
# #     x = x - x.mean(dim=-1, keepdim=True)
# #     y = y - y.mean(dim=-1, keepdim=True)
# #
# #     cov = (x * y).sum(dim=-1)
# #     std_x = torch.sqrt((x ** 2).sum(dim=-1) + eps)
# #     std_y = torch.sqrt((y ** 2).sum(dim=-1) + eps)
# #
# #     return cov / (std_x * std_y)
# #
# #
# #
# # # 示例
# # a = torch.tensor([0.1, 0.4, 0.0, 0.5])
# # b = torch.tensor([0.1, 0.45, 0.1, 0.35])
# # print("Pearson correlation:", pearson_corr(a, b).item())
#
import torch
import matplotlib.pyplot as plt

# # --- 1. 定义您的三个张量 ---
# teacher_ori = torch.tensor(
#        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0095, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0208, 0.1408, 0.0134, 0.0000],
#         [0.2378, 0.8207, 0.7201, 0.8016, 0.8912, 1.0056, 0.5652, 0.3621],
#         [0.6831, 2.0622, 1.6637, 1.5653, 1.4732, 1.5585, 1.0480, 0.7251],
#         [0.9085, 2.4354, 2.4923, 2.2444, 1.9250, 2.0339, 1.3363, 1.3076],
#         [0.8277, 1.7840, 2.1733, 1.9192, 1.8460, 1.9256, 1.6552, 1.2860],
#         [0.2937, 0.5377, 0.9561, 1.0636, 1.1240, 0.8901, 0.5986, 0.3521],
#         [0.1014, 0.2738, 0.3247, 0.1576, 0.1558, 0.1114, 0.3030, 0.3420]]
# )
#
# teacher_low = torch.tensor(
#        [[-0.0894, -0.0754, -0.0555, -0.0415, -0.0415, -0.0555, -0.0754, -0.0894],
#         [-0.0160,  0.0706,  0.1931,  0.2797,  0.2797,  0.1931,  0.0706, -0.0160],
#         [ 0.3126,  0.5018,  0.7694,  0.9586,  0.9586,  0.7694,  0.5018,  0.3126],
#         [ 0.9012,  1.1630,  1.5332,  1.7950,  1.7950,  1.5332,  1.1630,  0.9012],
#         [ 1.3786,  1.6404,  2.0106,  2.2724,  2.2724,  2.0106,  1.6404,  1.3786],
#         [ 1.3089,  1.4981,  1.7657,  1.9549,  1.9549,  1.7657,  1.4981,  1.3089],
#         [ 0.7062,  0.7928,  0.9153,  1.0019,  1.0019,  0.9153,  0.7928,  0.7062],
#         [ 0.1212,  0.1353,  0.1551,  0.1691,  0.1691,  0.1551,  0.1353,  0.1212]]
# )
#
# teacher_high = torch.tensor(
#        [[ 0.0894,  0.0754,  0.0555,  0.0415,  0.0415,  0.0650,  0.0754,  0.0894],
#         [ 0.0160, -0.0706, -0.1931, -0.2797, -0.2589, -0.0523, -0.0572,  0.0160],
#         [-0.0748,  0.3189, -0.0493, -0.1570, -0.0674,  0.2362,  0.0634,  0.0496],
#         [-0.2182,  0.8992,  0.1305, -0.2297, -0.3218,  0.0253, -0.1150, -0.1762],
#         [-0.4701,  0.7950,  0.4817, -0.0280, -0.3474,  0.0233, -0.3042, -0.0710],
#         [-0.4811,  0.2859,  0.4077, -0.0357, -0.1089,  0.1599,  0.1572, -0.0229],
#         [-0.4126, -0.2552,  0.0408,  0.0617,  0.1221, -0.0252, -0.1943, -0.3541],
#         [-0.0199,  0.1385,  0.1696, -0.0116, -0.0134, -0.0437,  0.1677,  0.2208]]
# )
#
# # --- 2. 确定两种颜色范围 ---
#
# # 范围 1：用于 Ori 和 Low (全局结构)
# # 使用 'viridis' (从暗到亮)
# cmap1 = 'viridis'
# vmin1 = min(teacher_ori.min(), teacher_low.min()).item() # 约 -0.09
# vmax1 = max(teacher_ori.max(), teacher_low.max()).item() # 约 2.49
#
# # 范围 2：用于 High (细节/边缘)
# # 使用 'coolwarm' (蓝-白-红)，以 0 为中心
# cmap2 = 'coolwarm'
# v_abs_max2 = teacher_high.abs().max().item() # 约 0.9
# vmin2 = -v_abs_max2
# vmax2 = v_abs_max2
#
# # --- 3. 可视化 ---
# # 创建一个 1x3 的 "画布" (Figure)，并设置足够的宽度
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
#
# # 图 1: 原始特征
# im1 = ax1.imshow(teacher_ori.cpu().numpy(),
#                  cmap=cmap1, vmin=vmin1, vmax=vmax1)
# ax1.set_title("Original Feature")
# fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04) # 添加色条
#
# # 图 2: 低频特征
# im2 = ax2.imshow(teacher_low.cpu().numpy(),
#                  cmap=cmap1, vmin=vmin1, vmax=vmax1)
# ax2.set_title("Low Freq Feature")
# fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04) # 添加色条
#
# # 图 3: 高频特征
# im3 = ax3.imshow(teacher_high.cpu().numpy(),
#                  cmap=cmap2, vmin=vmin2, vmax=vmax2)
# ax3.set_title("High Freq Feature")
# fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04) # 添加色条
#
# plt.tight_layout() # 自动调整布局
# plt.show()




# import torch
# import matplotlib.pyplot as plt
#
# # 1. 定义您的 DCT 特征图张量
# dct_x = torch.tensor(
#        [[ 0.3210, -0.0878, -0.1909,  0.1048,  0.0423, -0.0040, -0.0285,  0.0374],
#         [-0.4448,  0.1219,  0.2642, -0.1457, -0.0582,  0.0062,  0.0392, -0.0526],
#         [ 0.4176, -0.1152, -0.2470,  0.1383,  0.0534, -0.0075, -0.0362,  0.0514],
#         [-0.3740,  0.1042,  0.2199, -0.1260, -0.0463,  0.0089,  0.0316, -0.0488],
#         [ 0.3162, -0.0891, -0.1846,  0.1086,  0.0375, -0.0097, -0.0259,  0.0441],
#         [-0.2470,  0.0704,  0.1432, -0.0864, -0.0280,  0.0093,  0.0196, -0.0366],
#         [ 0.1693, -0.0487, -0.0975,  0.0602,  0.0185, -0.0074, -0.0130,  0.0264],
#         [-0.0860,  0.0249,  0.0493, -0.0309, -0.0091,  0.0041,  0.0065, -0.0139]]
# )
#
# # 2. 确定颜色范围
# # 我们使用一个以 0 为中心的对称范围，以便清晰地看到正负系数
# v_max = dct_x.abs().max().item() # 找到绝对值的最大值
# v_min = -v_max
#
# # 3. 创建画布和热力图
# plt.figure(figsize=(8, 6)) # 设置画布大小
# ax = plt.gca() # 获取当前轴
#
# # 使用 'coolwarm' 色图 (蓝-白-红)，并设置对称的颜色范围
# im = ax.imshow(dct_x.cpu().numpy(), cmap='coolwarm', vmin=v_min, vmax=v_max)
#
# # 设置标题
# ax.set_title("DCT Frequency Coefficients Heatmap")
#
# # 在每个格子上显示数值
# # (如果特征图很大，可以注释掉这部分)
# for i in range(dct_x.shape[0]):
#     for j in range(dct_x.shape[1]):
#         ax.text(j, i, f'{dct_x[i, j]:.2f}',
#                 ha='center', va='center', color='black', fontsize=8)
#
# # 添加颜色条 (Colorbar)
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
# # 显示图像
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
#
# # --- 1. 全局样式设置 (学术风格) ---
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],  # 论文标准字体
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 13,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 11,
#     "lines.linewidth": 2,
#     "figure.dpi": 150,   # 预览清晰度
#     # "text.usetex": True # 如果电脑安装了 LaTeX，取消注释这行，公式会更漂亮
# })
#
# # --- 2. 数据录入 (根据你提供的图片) ---
#
# # 数据组 1: 旋转角度 (Rotation Angle)
# # 对应图片: image_558f2a.png
# x_rot = [1, 3, 5, 7, 9]
# y_rot_origin = [0.88, 0.78, 0.73, 0.69, 0.65]
# y_rot_low    = [0.94, 0.90, 0.88, 0.87, 0.84]
# y_rot_high   = [0.80, 0.64, 0.57, 0.50, 0.41]
#
# # 数据组 2: 高斯噪声 (Gaussian Noise)
# # 对应图片: image_558fe1.png
# x_noise = [0.1, 0.2, 0.3, 0.4, 0.5]
# y_noise_origin = [0.87, 0.78, 0.71, 0.65, 0.60]
# y_noise_low    = [0.94, 0.89, 0.85, 0.82, 0.80]
# y_noise_high   = [0.83, 0.66, 0.54, 0.45, 0.38]
#
# # 数据组 3: 图像缩放 (Image Scaling)
# # 对应图片: image_558f84.png
# # 注意：X轴是 0.9 -> 0.1 (降采样程度变大)，为了绘图方便，我们按顺序绘图并替换标签
# x_scale_labels = [0.9, 0.7, 0.5, 0.3, 0.1]
# x_scale_idx    = range(len(x_scale_labels)) # [0, 1, 2, 3, 4] 用于绘图定位
# y_scale_origin = [0.91, 0.88, 0.82, 0.69, 0.46]
# y_scale_low    = [0.95, 0.94, 0.91, 0.88, 0.71]
# y_scale_high   = [0.86, 0.82, 0.72, 0.64, 0.19]
#
#
# # --- 3. 绘图逻辑 ---
# fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True) # 1行3列
#
# # 定义配色方案 (Color Palette)
# # Origin: 灰色虚线 (Baseline)
# # Low Freq: 红色/橙色 (代表鲁棒、能量集中)
# # High Freq: 蓝色/绿色 (代表敏感、易丢失)
# c_origin = '#A9D18E'
# c_low    = '#F8CBAD'  # Brick Red
# c_high   = '#8FAADC'  # Muted Blue
#
# # 定义标记形状 (Markers) - 保证黑白打印也能区分
# m_origin = 's' # Square (方块)
# m_low    = 'o' # Circle (圆点)
# m_high   = '^' # Triangle (三角)
#
# # --------------------------
# # 子图 1: Rotation Angle
# # --------------------------
# ax = axes[0]
# ax.plot(x_rot, y_rot_low,    color=c_low,    marker=m_low,    label='Low Freq (Robust)', linewidth=2.5, markersize=8,  zorder=3)
# ax.plot(x_rot, y_rot_origin, color=c_origin, marker=m_origin, label='Origin Feature',  linewidth=2.5, markersize=8,   linestyle='--', alpha=0.8, zorder=2)
# ax.plot(x_rot, y_rot_high,   color=c_high,   marker=m_high,   label='High Freq (Sensitive)', linewidth=2.5, markersize=8, zorder=3)
#
# # 填充区域 (Highlight Gap)
# ax.fill_between(x_rot, y_rot_low, y_rot_high, color='gray', alpha=0.05)
#
# ax.set_title("Rotation", fontweight='bold')
# ax.set_xlabel("Rotation Angle (degrees)")
# ax.set_ylabel("Cosine Similarity") # 只在最左侧显示Y轴标签
# ax.set_xticks(x_rot)
# ax.grid(True, linestyle=':', alpha=0.6)
# ax.set_ylim(0.1, 1.05) # 设置Y轴范围，留出顶部放图例的空间
#
# # --------------------------
# # 子图 2: Gaussian Noise
# # --------------------------
# ax = axes[1]
# ax.plot(x_noise, y_noise_low,    color=c_low,    marker=m_low,    zorder=3, linewidth=2.5, markersize=8, )
# ax.plot(x_noise, y_noise_origin, color=c_origin, marker=m_origin, linestyle='--', alpha=0.8, zorder=2, linewidth=2.5, markersize=8, )
# ax.plot(x_noise, y_noise_high,   color=c_high,   marker=m_high,   zorder=3, linewidth=2.5, markersize=8, )
#
# ax.fill_between(x_noise, y_noise_low, y_noise_high, color='gray', alpha=0.05)
#
# ax.set_title("Gaussian Noise", fontweight='bold')
# ax.set_xlabel("Noise Intensity ($\sigma$)")
# ax.set_xticks(x_noise)
# ax.grid(True, linestyle=':', alpha=0.6)
#
# # --------------------------
# # 子图 3: Image Scaling
# # --------------------------
# ax = axes[2]
# ax.plot(x_scale_idx, y_scale_low,    color=c_low,    marker=m_low,    zorder=3, linewidth=2.5, markersize=8, )
# ax.plot(x_scale_idx, y_scale_origin, color=c_origin, marker=m_origin, linestyle='--', alpha=0.8, zorder=2, linewidth=2.5, markersize=8, )
# ax.plot(x_scale_idx, y_scale_high,   color=c_high,   marker=m_high,   zorder=3, linewidth=2.5, markersize=8, )
#
# ax.fill_between(x_scale_idx, y_scale_low, y_scale_high, color='gray', alpha=0.05)
#
# ax.set_title("Image Scaling", fontweight='bold')
# ax.set_xlabel("Scaling Factor")
# # 替换X轴刻度：显示原始的 0.9 -> 0.1
# ax.set_xticks(x_scale_idx)
# ax.set_xticklabels([str(x) for x in x_scale_labels])
# ax.grid(True, linestyle=':', alpha=0.6)
#
# # --- 4. 图例与布局调整 ---
# # 获取第一个子图的图例句柄，统一放置在顶部居中
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
#
# plt.tight_layout()
#
# # 保存命令 (建议保存为 PDF 放入 LaTeX)
# # plt.savefig('feature_sensitivity_analysis.pdf', bbox_inches='tight', dpi=300)
#
# plt.show()

#####################################################################################################################

# import matplotlib.pyplot as plt
# import numpy as np
#
# # --- 1. 全局样式设置 ---
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 13,
#     "xtick.labelsize": 10, # 稍微调小一点以防X轴标签重叠
#     "ytick.labelsize": 11,
#     "legend.fontsize": 12,
#     "lines.linewidth": 2,
#     "figure.dpi": 150,
# })
#
# # --- 2. 数据录入 ---
#
# # [新增] 数据组 0: 跨架构特征一致性 (Bar Chart 数据)
# # 估算自原图 (a) Feature Consistency
# bar_labels = ['ResNet34\nResNet18', 'VGG19\nVGG11', 'VGG19\nResNet18'] # 换行以节省空间
# bar_x = np.arange(len(bar_labels))
# bar_origin = [0.38, 0.16, 0.22]
# bar_low    = [0.66, 0.52, 0.56]
# bar_high   = [0.06, 0.03, 0.03]
#
# # 数据组 1: 旋转 (Rotation)
# x_rot = [1, 3, 5, 7, 9]
# y_rot_origin = [0.88, 0.78, 0.73, 0.69, 0.65]
# y_rot_low    = [0.94, 0.90, 0.88, 0.87, 0.84]
# y_rot_high   = [0.80, 0.64, 0.57, 0.50, 0.41]
#
# # 数据组 2: 噪声 (Noise)
# x_noise = [0.1, 0.2, 0.3, 0.4, 0.5]
# y_noise_origin = [0.87, 0.78, 0.71, 0.65, 0.60]
# y_noise_low    = [0.94, 0.89, 0.85, 0.82, 0.80]
# y_noise_high   = [0.83, 0.66, 0.54, 0.45, 0.38]
#
# # 数据组 3: 缩放 (Scaling)
# x_scale_labels = [0.9, 0.7, 0.5, 0.3, 0.1]
# x_scale_idx    = range(len(x_scale_labels))
# y_scale_origin = [0.91, 0.88, 0.82, 0.69, 0.46]
# y_scale_low    = [0.95, 0.94, 0.91, 0.88, 0.71]
# y_scale_high   = [0.86, 0.82, 0.72, 0.64, 0.19]
#
# # --- 3. 绘图逻辑 (1行4列) ---
# # figsize=(16, 3.5): 总宽16，高3.5。单图宽度约 3.x 英寸，略呈长方形，不会太宽。
# fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
#
# # 统一配色与标记
# c_origin = '#A9D18E'
# c_low    = '#F8CBAD'
# c_high   = '#8FAADC'
# m_origin = 's'
# m_low    = 'o'
# m_high   = '^'
#
# # --------------------------
# # 图 1: Feature Consistency (Bar Chart)
# # --------------------------
# ax = axes[0]
# width = 0.25 # 柱子宽度
#
# # 绘制柱状图
# r1 = bar_x - width
# r2 = bar_x
# r3 = bar_x + width
#
# ax.bar(r1, bar_origin, color=c_origin, width=width, label='Origin Feature', edgecolor='white')
# ax.bar(r2, bar_low,    color=c_low,    width=width, label='Low Frequency',  edgecolor='white')
# ax.bar(r3, bar_high,   color=c_high,   width=width, label='High Frequency', edgecolor='white')
#
# ax.set_title("(a) Consistency", fontweight='bold')
# ax.set_xticks(bar_x)
# ax.set_xticklabels(bar_labels)
# ax.set_ylabel("Cosine Similarity") # 最左侧显示Y轴标签
# ax.grid(axis='y', linestyle=':', alpha=0.6)
# ax.set_ylim(0, 1.05) # 统一Y轴范围
#
# # --------------------------
# # 图 2: Rotation (Line Chart)
# # --------------------------
# ax = axes[1]
# ax.plot(x_rot, y_rot_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
# ax.plot(x_rot, y_rot_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
# ax.plot(x_rot, y_rot_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
# ax.fill_between(x_rot, y_rot_low, y_rot_high, color='gray', alpha=0.1)
#
# ax.set_title("(b) Rotation", fontweight='bold')
# ax.set_xlabel("Angle (deg)")
# ax.set_xticks(x_rot)
# ax.grid(True, linestyle=':', alpha=0.6)
#
# # --------------------------
# # 图 3: Gaussian Noise (Line Chart)
# # --------------------------
# ax = axes[2]
# ax.plot(x_noise, y_noise_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
# ax.plot(x_noise, y_noise_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
# ax.plot(x_noise, y_noise_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
# ax.fill_between(x_noise, y_noise_low, y_noise_high, color='gray', alpha=0.1)
#
# ax.set_title("(c) Noise", fontweight='bold')
# ax.set_xlabel("Intensity ($\sigma$)")
# ax.set_xticks(x_noise)
# ax.grid(True, linestyle=':', alpha=0.6)
#
# # --------------------------
# # 图 4: Scaling (Line Chart)
# # --------------------------
# ax = axes[3]
# ax.plot(x_scale_idx, y_scale_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
# ax.plot(x_scale_idx, y_scale_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
# ax.plot(x_scale_idx, y_scale_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
# ax.fill_between(x_scale_idx, y_scale_low, y_scale_high, color='gray', alpha=0.1)
#
# ax.set_title("(d) Scaling", fontweight='bold')
# ax.set_xlabel("Factor")
# ax.set_xticks(x_scale_idx)
# ax.set_xticklabels([str(x) for x in x_scale_labels])
# ax.grid(True, linestyle=':', alpha=0.6)
#
# # --- 4. 统一图例 ---
# # 由于柱状图和折线图含义一致，我们取第一个图的图例即可
# handles, labels = axes[0].get_legend_handles_labels()
# # 放在整张图的上方居中
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
#
# # 调整布局，避免重叠
# # w_space=0.15 控制子图之间的横向间距
# plt.subplots_adjust(wspace=0.15, left=0.05, right=0.99, bottom=0.15, top=0.85)
#
# # 保存时自动裁剪白边
# plt.savefig('combined_analysis.png', bbox_inches='tight', dpi=300)
#
# plt.show()

##########################################################################

import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局样式设置 ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "figure.dpi": 150,
})

# --- 2. 数据录入 ---

# [新增] 数据组 0: 跨架构特征一致性 (Bar Chart 数据)
bar_labels = ['ResNet34\nResNet18', 'VGG19\nVGG11', 'VGG19\nResNet18']
bar_x = np.arange(len(bar_labels))
# bar_origin = [0.38, 0.16, 0.22]
# bar_low    = [0.66, 0.52, 0.56]
# bar_high   = [0.06, 0.03, 0.03]
bar_origin = [0.38, 0.19, 0.26]
bar_low    = [0.57, 0.47, 0.50]
bar_high   = [0.046, 0.041, 0.026]

# 数据组 1: 旋转 (Rotation)
x_rot = [1, 3, 5, 7, 9]
y_rot_origin = [0.88, 0.78, 0.73, 0.69, 0.65]
y_rot_low    = [0.93, 0.87, 0.84, 0.81, 0.79]
y_rot_high   = [0.80, 0.64, 0.56, 0.48, 0.43]

# 数据组 2: 噪声 (Noise)
x_noise = [0.1, 0.2, 0.3, 0.4, 0.5]
y_noise_origin = [0.87, 0.78, 0.71, 0.65, 0.60]
y_noise_low    = [0.93, 0.86, 0.81, 0.76, 0.73]
y_noise_high   = [0.82, 0.66, 0.54, 0.44, 0.37]

# 数据组 3: 缩放 (Scaling)
x_scale_labels = [0.9, 0.7, 0.5, 0.3, 0.1]
x_scale_idx    = range(len(x_scale_labels))
y_scale_origin = [0.91, 0.88, 0.82, 0.69, 0.46]
y_scale_low    = [0.94, 0.93, 0.88, 0.80, 0.62]
y_scale_high   = [0.86, 0.82, 0.72, 0.52, 0.19]

# --- 3. 绘图逻辑 (1行4列) ---
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)

# 统一配色与标记
c_origin = '#A9D18E'
c_low    = '#F8CBAD'
c_high   = '#8FAADC'
m_origin = 's'
m_low    = 'o'
m_high   = '^'

# --------------------------
# 图 1: Cross-Architecture Alignment (Bar Chart)
# --------------------------
ax = axes[0]
width = 0.25

r1 = bar_x - width
r2 = bar_x
r3 = bar_x + width

ax.bar(r1, bar_origin, color=c_origin, width=width, label='Origin Feature', edgecolor='white')
ax.bar(r2, bar_low,    color=c_low,    width=width, label='Low Frequency Feature',  edgecolor='white')
ax.bar(r3, bar_high,   color=c_high,   width=width, label='High Frequency Feature', edgecolor='white')

# Title 方案一：学术风
ax.set_title("(a) Networks Consistency", fontweight='bold', fontsize=13)
ax.set_xticks(bar_x)
ax.set_xticklabels(bar_labels)
ax.set_ylabel("Cosine Similarity")
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.set_ylim(0, 1.05)

# --------------------------
# 图 2: Robustness to Rotation
# --------------------------
ax = axes[1]
ax.plot(x_rot, y_rot_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
ax.plot(x_rot, y_rot_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
ax.plot(x_rot, y_rot_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
ax.fill_between(x_rot, y_rot_low, y_rot_high, color='gray', alpha=0.1)

# Title 方案一
ax.set_title("(b) Rotation", fontweight='bold', fontsize=13)
ax.set_xlabel("Rotation Angle")
ax.set_xticks(x_rot)
ax.grid(True, linestyle=':', alpha=0.6)

# --------------------------
# 图 3: Robustness to Gaussian Noise
# --------------------------
ax = axes[2]
ax.plot(x_noise, y_noise_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
ax.plot(x_noise, y_noise_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
ax.plot(x_noise, y_noise_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
ax.fill_between(x_noise, y_noise_low, y_noise_high, color='gray', alpha=0.1)

# Title 方案一
ax.set_title("(c) Gaussian Noise", fontweight='bold', fontsize=13)
ax.set_xlabel("Noise Intensity ($\sigma$)")
ax.set_xticks(x_noise)
ax.grid(True, linestyle=':', alpha=0.6)

# --------------------------
# 图 4: Robustness to Image Scaling
# --------------------------
ax = axes[3]
ax.plot(x_scale_idx, y_scale_low,    color=c_low,    marker=m_low,    linewidth=2.5, markersize=7)
ax.plot(x_scale_idx, y_scale_origin, color=c_origin, marker=m_origin, linewidth=2.5, markersize=7, linestyle='--', alpha=0.9)
ax.plot(x_scale_idx, y_scale_high,   color=c_high,   marker=m_high,   linewidth=2.5, markersize=7)
ax.fill_between(x_scale_idx, y_scale_low, y_scale_high, color='gray', alpha=0.1)

# Title 方案一
ax.set_title("(d) Image Scaling", fontweight='bold', fontsize=13)
ax.set_xlabel("Scaling Factor")
ax.set_xticks(x_scale_idx)
ax.set_xticklabels([str(x) for x in x_scale_labels])
ax.grid(True, linestyle=':', alpha=0.6)

# --- 4. 统一图例 ---
handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)

plt.subplots_adjust(wspace=0.15, left=0.05, right=0.99, bottom=0.25, top=0.9)

# Save as PNG with minimal whitespace
plt.savefig('combined_analysis.png', bbox_inches='tight', dpi=300)

