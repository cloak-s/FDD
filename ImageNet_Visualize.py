import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import cifar_model_dict, imagenet_model_dict
from utils import accuracy, AverageMeter, train_epoch, validate, get_cifar_dataset, log_msg, setup_logger, set_seed
import torch.nn.functional as F


def create_imagenet_loaders(data_root, batch_size=128, num_workers=8, input_size=224):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # 2. 定义路径
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # 简单路径检查
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"警告：在 '{data_root}' 中未找到 'train' 或 'val' 文件夹。")

    # 3. 实例化变换 (直接内联)
    print(f'训练集 input_size: {input_size}')
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 验证集通常固定为224
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 4. 创建 Datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    # 5. 创建 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_pearson_correlation(x, y):
    # x = x - x.mean(dim=1, keepdim=True)
    # y = y - y.mean(dim=1, keepdim=True)

    corr = F.cosine_similarity(x, y, dim=-1)

    return corr


import random

def get_decouple_feature(x, mask_ratio=[0.05, 0.08, 0.10]):
    B, C, H, W = x.shape

    # 去中心化
    # x = x - x.mean(dim=(-2, -1), keepdim=True)

    mask_ratio = mask_ratio if isinstance(mask_ratio, (float, int)) else random.choice(mask_ratio)

    spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
    magnitude = spectrum.abs()  #(B, C, H, W//2 + 1)

    magnitude_mean = torch.mean(magnitude, dim=1, keepdim=True) # #(B, 1, H, W//2 + 1)
    magnitude_flat = magnitude_mean.view(B, -1)
    threshold = torch.quantile(magnitude_flat, q= 1 - mask_ratio, dim=-1, keepdim=True)  # B * 1

    threshold = threshold.unsqueeze(-1).unsqueeze(-1)
    mask = (magnitude_mean >= threshold)  # (B, 1, H, W_fft)  W_fft = W // 2 + 1

    recon_x = torch.where(mask, spectrum, 0.)
    com_x = torch.fft.irfft2(recon_x, s=(H, W), dim=(-2, -1), norm='ortho')
    r_com = x - com_x
    return mask, com_x, r_com


def get_fixed_radius_decouple(x, radius=1.5):
    x_fft = torch.fft.fft2(x, norm='ortho', dim=(-2, -1))
    x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # [B,C,H,W] complex

    B, C, H, W = x_fft.shape
    device = x.device

    # Build distance grid (H,W) with DC at center
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=device) - cy
    xx = torch.arange(W, device=device) - cx
    Y, X = torch.meshgrid(yy, xx, indexing='ij')  # [H,W]

    dist2 = (Y.to(torch.float32) ** 2 + X.to(torch.float32) ** 2)
    r2 = float(radius) ** 2

    low_mask = (dist2 <= r2).view(1, 1, H, W)  # broadcast to [B,C,H,W]

    x_low_fft = x_fft * low_mask
    # x_high_fft = x_fft * (~low_mask)  # boolean mask is fine; will upcast

    # Inverse: shift back -> iFFT
    x_low = torch.fft.ifft2(torch.fft.ifftshift(x_low_fft, dim=(-2, -1)),
                            norm='ortho', dim=(-2, -1)).real
    x_high = x - x_low
    return low_mask, x_low, x_high


def get_student_decouple_feat(x, mask):
    B, C, H, W = x.shape

    # 去中心化
    # x = x - x.mean(dim=(-2, -1), keepdim=True)

    fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
    recon_x = fft_x * mask
    com_x = torch.fft.irfft2(recon_x, s=(H, W), dim=(-2, -1), norm='ortho')
    r_com = x - com_x
    return com_x, r_com



# import torch
#
#
#
# if __name__ == '__main__':
#     set_seed(42)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
#
#     student_model = imagenet_model_dict['vgg11'](num_classes=1000)
#     teacher_model = imagenet_model_dict['vgg19'](num_classes=1000)
#     # teacher_model = imagenet_model_dict['ResNet34'](num_classes=1000)
#     # student_model = imagenet_model_dict['ResNet18'](num_classes=1000)
#
#     # pretrained_teacher = torch.load('Pretrain/ImageNet/ResNet34_vanilla/ckpt_resnet34.pth')
#     # pretrained_student = torch.load('Pretrain/ImageNet/ResNet18_vanilla/ckpt_resnet18.pth')
#     # pretrained_student = torch.load('Pretrain/ImageNet/ResNet18_FreDKD/student_best')
#
#     pretrained_teacher = torch.load('Pretrain/ImageNet/vgg19_bn_vanilla/vgg19_bn-c79401a0.pth')
#     pretrained_student = torch.load('Pretrain/ImageNet/vgg11_bn_vanilla/vgg11_bn-6002323d.pth')
#     # pretrained_student = torch.load('Pretrain/ImageNet/ResNet18_vanilla/ckpt_resnet18.pth')
#     # pretrained_student = torch.load('Pretrain/ImageNet/vgg11_bn_vanilla/vgg11_bn-6002323d.pth')
#
#     teacher_model.load_state_dict(pretrained_teacher)
#     student_model.load_state_dict(pretrained_student)
#
#     teacher_model.eval()
#     student_model.eval()
#
#     teacher_model.cuda()
#     student_model.cuda()
#
#     train_loader, val_loader = create_imagenet_loaders(data_root='/data1/ImageNet2012', batch_size=512, num_workers=8, input_size=224)
#
#     images, labels = next(iter(val_loader))
#     print(f"  - 图像 (Images) 的形状 (Shape): {images.shape}")
#
#     images = images.cuda()
#     labels = labels.cuda()
#
#     with torch.no_grad():
#         t_logit, t_feats = teacher_model(images)
#         s_logit, s_feats = student_model(images)
#
#     print('f_t 的个数', len(t_feats['feats']))
#     print('f_s 的个数', len(s_feats['feats']))
#     f_t = t_feats['feats'][-1]
#     f_s = s_feats['feats'][-1]
#     print(f"  - <teacher> (Features): {f_t.shape}")
#     print(f"  - <student> (Features): {f_s.shape}")
#
#     t_mask, t_low_feat, t_high_feat = get_decouple_feature(f_t, mask_ratio=[0.10])
#     # s_mask, s_low_feat, s_high_feat = get_decouple_feature(f_s, mask_ratio=[0.1])
#     s_low_feat, s_high_feat = get_student_decouple_feat(f_s, t_mask)
#
#     # t_mask, t_low_feat, t_high_feat = get_fixed_radius_decouple(f_t, radius=1.0)
#     # s_mask, s_low_feat, s_high_feat = get_fixed_radius_decouple(f_s, radius=1.0)
#
#
#     # # B * C * H * W
#     # B, C, H, W = f_t.shape
#     # L = H * W  # L = H * W
#     #
#     # print("\n--- (新方法) 开始计算皮尔逊相关性 (通道平均后) ---")
#     #
#     # # 1. 原始特征相关性
#     # # 形状从 [B, C, H, W] -> [B, C, L]
#     # f_t_3d = f_t.reshape(B, C, L)
#     # f_s_3d = f_s.reshape(B, C, L)
#     #
#     # # 在通道维度(dim=1)上求平均 -> [B, L]
#     # f_t_mean_c = torch.mean(f_t_3d, dim=1)
#     # f_s_mean_c = torch.mean(f_s_3d, dim=1)
#     #
#     # # get_pearson_correlation(x[B, L], y[B, L]) 将在 dim=1 (即 L 维度)上计算
#     # # 返回一个 [B] 的张量，表示每个样本的相关性
#     # corr_orig = get_pearson_correlation(f_t_mean_c, f_s_mean_c)
#     # # 计算批次中所有样本的平均相关性
#     # mean_corr_orig = corr_orig.mean()
#     #
#     # print(f"1. 原始特征 (Reshaped to {f_t_mean_c.shape} after channel avg):")
#     # print(f"   平均相关性: {mean_corr_orig.item():.4f}")
#     #
#     # # 2. 低频特征相关性
#     # # 形状 [B, C, H, W] -> [B, C, L] -> [B, L] (在dim=1上平均)
#     # t_low_mean_c = torch.mean(t_low_feat.reshape(B, C, L), dim=1)
#     # s_low_mean_c = torch.mean(s_low_feat.reshape(B, C, L), dim=1)
#     #
#     # # corr_low 形状为 [B]
#     # corr_low = get_pearson_correlation(t_low_mean_c, s_low_mean_c)
#     # mean_corr_low = corr_low.mean()
#     #
#     # print(f"2. 低频特征 (Reshaped to {t_low_mean_c.shape} after channel avg):")
#     # print(f"   平均相关性: {mean_corr_low.item():.4f}")
#     #
#     # # 3. 高频特征相关性
#     # # 形状 [B, C, H, W] -> [B, C, L] -> [B, L] (在dim=1上平均)
#     # t_high_mean_c = torch.mean(t_high_feat.reshape(B, C, L), dim=1)
#     # s_high_mean_c = torch.mean(s_high_feat.reshape(B, C, L), dim=1)
#     #
#     # # corr_high 形状为 [B]
#     # corr_high = get_pearson_correlation(t_high_mean_c, s_high_mean_c)
#     # mean_corr_high = corr_high.mean()
#     #
#     # print(f"3. 高频特征 (Reshaped to {t_high_mean_c.shape} after channel avg):")
#     # print(f"   平均相关性: {mean_corr_high.item():.4f}")
#
#     # B * C * H * W
#     B, C, H, W = f_t.shape
#     L = H * W  # L = H * W
#
#     # ==========================================================
#     # ⬇️ 补充的代码从这里开始
#     # ==========================================================
#
#     print("\n--- 开始计算相关性 ---")
#
#     # 1. 原始特征相关性
#     # 形状从 [B, C, H, W] -> [B*C, H*W]
#     f_t_reshaped = f_t.reshape(B * C, L)
#     f_s_reshaped = f_s.reshape(B * C, L)
#
#     # corr_orig 形状为 [B*C]
#     corr_orig = get_pearson_correlation(f_t_reshaped, f_s_reshaped)
#     # 计算所有样本和通道的平均相关性
#     mean_corr_orig = corr_orig.mean()
#
#     print(f"1. 原始特征 (Reshaped to {f_t_reshaped.shape}):")
#     print(f"   平均相关性: {mean_corr_orig.item():.4f}")
#
#     # 2. 低频特征相关性
#     # 形状从 [B, C, H, W] -> [B*C, H*W]
#     t_low_reshaped = t_low_feat.reshape(B * C, L)
#     s_low_reshaped = s_low_feat.reshape(B * C, L)
#
#     # corr_low 形状为 [B*C]
#     corr_low = get_pearson_correlation(t_low_reshaped, s_low_reshaped)
#     mean_corr_low = corr_low.mean()
#
#     print(f"2. 低频特征 (Reshaped to {t_low_reshaped.shape}):")
#     print(f"   平均相关性: {mean_corr_low.item():.4f}")
#
#     # 3. 高频特征相关性
#     # 形状从 [B, C, H, W] -> [B*C, H*W]
#     t_high_reshaped = t_high_feat.reshape(B * C, L)
#     s_high_reshaped = s_high_feat.reshape(B * C, L)
#
#     # corr_high 形状为 [B*C]
#     corr_high = get_pearson_correlation(t_high_reshaped, s_high_reshaped)
#     mean_corr_high = corr_high.mean()
#
#     print(f"3. 高频特征 (Reshaped to {t_high_reshaped.shape}):")
#     print(f"   平均相关性: {mean_corr_high.item():.4f}")





    # # --- Teacher Energy ---
    # E_t_total = torch.norm(f_t, p=2) ** 2
    # E_t_low = torch.norm(t_low_feat, p=2) ** 2
    # E_t_high = torch.norm(t_high_feat, p=2) ** 2
    #
    # # --- Student Energy ---
    # E_s_total = torch.norm(f_s, p=2) ** 2
    # E_s_low = torch.norm(s_low_feat, p=2) ** 2
    # E_s_high = torch.norm(s_high_feat, p=2) ** 2
    #
    # # --- Ratios ---
    # # 为了防止除零错误，加一个极小值 epsilon (虽然通常不会为0)
    # eps = 1e-8
    # ratio_t_low = E_t_low / (E_t_total + eps)
    # ratio_t_high = E_t_high / (E_t_total + eps)
    #
    # ratio_s_low = E_s_low / (E_s_total + eps)
    # ratio_s_high = E_s_high / (E_s_total + eps)
    #
    # print("\n" + "=" * 40)
    # print("   能量分布分析 (Energy Distribution Analysis)")
    # print("=" * 40)
    #
    # print(f"\n[Teacher Model - ResNet34]")
    # print(f"  Total Energy:               {E_t_total:.4e}")
    # print(f"  Low-Freq Energy (Main):     {E_t_low:.4e}  (占比: {ratio_t_low:.2%})")
    # print(f"  High-Freq Energy (Detail):  {E_t_high:.4e}  (占比: {ratio_t_high:.2%})")
    # print(f"  >> 能量守恒检查 (Low+High):   {(E_t_low + E_t_high):.4e} (理论上应接近 Total)")
    #
    # print(f"\n[Student Model - ResNet18]")
    # print(f"  Total Energy:               {E_s_total:.4e}")
    # print(f"  Low-Freq Energy (Main):     {E_s_low:.4e}  (占比: {ratio_s_low:.2%})")
    # print(f"  High-Freq Energy (Detail):  {E_s_high:.4e}  (占比: {ratio_s_high:.2%})")
    #
    # print("\n[对比总结 - Comparison]")
    # print(f"  Teacher 高频占比: {ratio_t_high:.2%}  vs  Student 高频占比: {ratio_s_high:.2%}")
    #
    # diff_high = ratio_t_high - ratio_s_high
    # if diff_high > 0:
    #     print(f"  => 结论: Teacher 包含比 Student 更多的 高频细节 (差值: +{diff_high * 100:.2f}%)")
    # else:
    #     print(f"  => 结论: Student 的高频噪声/细节 占比反常地高于 Teacher (差值: {diff_high * 100:.2f}%)")





# ******************************************************************************************




import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from models import cifar_model_dict, imagenet_model_dict
from utils import setup_logger, set_seed
import torch.nn.functional as F
import random


# ==========================================
# 1. 基础函数保持不变
# ==========================================

def create_imagenet_loaders(data_root, batch_size=128, num_workers=8, input_size=224):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dir = os.path.join(data_root, 'val')
    if not os.path.isdir(val_dir):
        print(f"警告：在 '{data_root}' 中未找到 'val' 文件夹。")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return None, val_loader  # 我们只需要验证集


def get_decouple_feature(x, mask_ratio=0.1):
    # 注意：为了保证对比公平，这里去掉了 random.choice，强制使用固定的 ratio
    B, C, H, W = x.shape

    # rfft2: 实数FFT
    spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
    magnitude = spectrum.abs()

    # 计算阈值
    magnitude_mean = torch.mean(magnitude, dim=1, keepdim=True)
    magnitude_flat = magnitude_mean.view(B, -1)
    threshold = torch.quantile(magnitude_flat, q=1 - mask_ratio, dim=-1, keepdim=True)
    threshold = threshold.unsqueeze(-1).unsqueeze(-1)

    # 生成掩码
    mask = (magnitude_mean >= threshold)

    # 1. 低频部分 (保留主要能量)
    recon_low = torch.where(mask, spectrum, torch.tensor(0., device=x.device))
    low_freq_x = torch.fft.irfft2(recon_low, s=(H, W), dim=(-2, -1), norm='ortho')

    # 2. 高频部分 (残差)
    high_freq_x = x - low_freq_x

    return low_freq_x, high_freq_x

def get_fixed_radiurs_feature(x, radius=1.0):
    x_fft = torch.fft.fft2(x, norm='ortho', dim=(-2, -1))
    x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # [B,C,H,W] complex

    B, C, H, W = x_fft.shape
    device = x.device

    # Build distance grid (H,W) with DC at center
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=device) - cy
    xx = torch.arange(W, device=device) - cx
    Y, X = torch.meshgrid(yy, xx, indexing='ij')  # [H,W]

    dist2 = (Y.to(torch.float32) ** 2 + X.to(torch.float32) ** 2)
    r2 = float(radius) ** 2

    low_mask = (dist2 <= r2).view(1, 1, H, W)  # broadcast to [B,C,H,W]

    x_low_fft = x_fft * low_mask
    # x_high_fft = x_fft * (~low_mask)  # boolean mask is fine; will upcast

    # Inverse: shift back -> iFFT
    x_low = torch.fft.ifft2(torch.fft.ifftshift(x_low_fft, dim=(-2, -1)),
                            norm='ortho', dim=(-2, -1)).real
    x_high = x - x_low
    return x_low, x_high




# ==========================================
# 2. 主程序 - 添加自对比验证
# ==========================================

if __name__ == '__main__':
    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 加载 Teacher 模型
    print(">>> 正在加载模型...")
    teacher_model = imagenet_model_dict['ResNet34'](num_classes=1000)
    pretrained_teacher = torch.load('Pretrain/ImageNet/ResNet34_vanilla/ckpt_resnet34.pth')
    teacher_model.load_state_dict(pretrained_teacher)
    teacher_model.eval().cuda()

    # 加载数据
    _, val_loader = create_imagenet_loaders(data_root='/data1/ImageNet2012', batch_size=512, num_workers=4)

    # 获取一个 Batch 的数据
    images_origin, _ = next(iter(val_loader))
    images_origin = images_origin.cuda()

    print(f"\n>>> 开始自对比实验 (Self-Consistency Experiment)")
    print(f"    Input Image Shape: {images_origin.shape}")

    # -------------------------------------------------
    # A. 构造空间扰动样本 (Shifted Images)
    # -------------------------------------------------
    # 我们对图片进行平移。
    # ResNet34 的下采样倍数是 32 (input 224 -> feat 7)。
    # 为了让特征图发生实质性错位，我们需要平移 input。
    # 平移 16 像素大约对应特征图上的 0.5 个像素相位差，足以破坏高频对齐。
    # shift_pixels = 16
    # print(f"    应用空间平移: {shift_pixels} pixels (模拟微小视角变化)")

    ############## 图像旋转
    # 使用 roll 进行循环移位 (或者用 padding 填充)
    # images_shifted = torch.roll(images_origin, shifts=(shift_pixels, shift_pixels), dims=(2, 3))

    angle = 9
    print(f"    应用旋转增强: {angle} degrees")

    # 使用 TF.rotate 进行旋转
    # expand=False 保持原图大小，由此产生的黑边对中心物体影响较小
    images_noisy = TF.rotate(images_origin, angle=angle)

    ############## 图像加噪
    # noise_sigma = 0.1
    # print(f"    应用高斯噪声: Sigma = {noise_sigma}")
    # print(f"    (这将向图像注入全频段干扰，重点测试高频特征的鲁棒性)")
    #
    # noise = torch.randn_like(images_origin, device=images_origin.device) * noise_sigma
    #
    # # 2. 叠加噪声
    # images_noisy = images_origin + noise

    # scale_ratio = 0.1
    # print(f"    应用图像缩放扰动: Ratio = {scale_ratio}")
    # print(f"    (操作: 下采样至 {scale_ratio * 100}% 尺寸，再双线性插值回原分辨率)")
    # print(f"    (原理: 模拟分辨率降低，直接导致高频信息因采样不足而永久丢失)")
    #
    # # 获取原始尺寸
    # H, W = images_origin.shape[2], images_origin.shape[3]
    #
    # # 1. 下采样 (Downsample) - 这是丢失信息的步骤
    # # 使用双线性插值 (bilinear) 或双三次插值 (bicubic)
    # images_down = F.interpolate(
    #     images_origin,
    #     scale_factor=scale_ratio,
    #     mode='bilinear',
    #     align_corners=False
    # )
    #
    # # 2. 上采样回原尺寸 (Upsample) - 重建图像以匹配网络输入
    # # 此时图像虽然尺寸恢复了，但高频细节已经变成了平滑的模糊
    # images_noisy = F.interpolate(
    #     images_down,
    #     size=(H, W),
    #     mode='bilinear',
    #     align_corners=False
    # )


    # -------------------------------------------------
    # B. 提取特征
    # -------------------------------------------------
    with torch.no_grad():
        _, t_feats_origin = teacher_model(images_origin)
        _, t_feats_shifted = teacher_model(images_noisy)

    # 取最后一层特征
    f_origin = t_feats_origin['feats'][4]  # [B, C, 7, 7]
    f_shifted = t_feats_shifted['feats'][4]

    B, C, H, W = f_origin.shape
    L = H * W
    print(f"    Feature Map Shape: {f_origin.shape}")

    # -------------------------------------------------
    # C. 频域解耦 (使用相同的 Mask Ratio)
    # -------------------------------------------------
    RATIO = 0.1
    low_origin, high_origin = get_decouple_feature(f_origin, mask_ratio=RATIO)
    low_shifted, high_shifted = get_decouple_feature(f_shifted, mask_ratio=RATIO)
    # low_origin, high_origin = get_fixed_radiurs_feature(f_origin, radius=1.0)
    # low_shifted, high_shifted = get_fixed_radiurs_feature(f_shifted, radius=1.0)

    # t_mask, t_low_feat, t_high_feat = get_decouple_feature(f_t, mask_ratio=[0.10])
    # # s_mask, s_low_feat, s_high_feat = get_decouple_feature(f_s, mask_ratio=[0.1])
    # s_low_feat, s_high_feat = get_student_decouple_feat(f_s, t_mask)

    # -------------------------------------------------
    # D. 计算相似度 (Validation)
    # -------------------------------------------------
    # 将特征展平为 [B*C, L] 来计算空间上的点对点相似度
    # 注意：我们关注的是"空间对齐"，所以要在 spatial dimension 上看是否 match

    # 1. 原始特征相似度
    sim_orig = F.cosine_similarity(f_origin.reshape(B * C, L), f_shifted.reshape(B * C, L), dim=1).mean()

    # 2. 低频特征相似度
    sim_low = F.cosine_similarity(low_origin.reshape(B * C, L), low_shifted.reshape(B * C, L), dim=1).mean()

    # 3. 高频特征相似度
    sim_high = F.cosine_similarity(high_origin.reshape(B * C, L), high_shifted.reshape(B * C, L), dim=1).mean()

    print("\n" + "=" * 50)
    print("   实验结果分析: Teacher vs Teacher (Shifted)")
    print("=" * 50)
    print(f"设定: 同一个网络，输入图像缩放规模 {angle} ")
    print("-" * 30)
    print(f"1. 原始特征相似度 (Original):  {sim_orig:.4f}")
    print(f"2. 低频特征相似度 (Low-Freq):  {sim_low:.4f} ")
    print(f"3. 高频特征相似度 (High-Freq): {sim_high:.4f} ")

    # -------------------------------------------------
    # E. 自动结论生成
    # -------------------------------------------------
    print("-" * 30)
    drop = sim_low - sim_high
    print(f"差异 (Low - High): {drop:.4f}")

