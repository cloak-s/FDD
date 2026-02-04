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

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"警告：在 '{data_root}' 中未找到 'train' 或 'val' 文件夹。")

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
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader



import random

def get_decouple_feature(x, mask_ratio=[0.05, 0.08, 0.10]):
    B, C, H, W = x.shape

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
#
#     images = images.cuda()
#     labels = labels.cuda()
#
#     with torch.no_grad():
#         t_logit, t_feats = teacher_model(images)
#         s_logit, s_feats = student_model(images)
#
#     f_t = t_feats['feats'][-1]
#     f_s = s_feats['feats'][-1]
#     print(f"  - <teacher> (Features): {f_t.shape}")
#     print(f"  - <student> (Features): {f_s.shape}")
#
#     t_mask, t_low_feat, t_high_feat = get_decouple_feature(f_t, mask_ratio=[0.10])
#     # s_mask, s_low_feat, s_high_feat = get_decouple_feature(f_s, mask_ratio=[0.1])
#     s_low_feat, s_high_feat = get_student_decouple_feat(f_s, t_mask)



#     f_t_reshaped = f_t.reshape(B * C, L)
#     f_s_reshaped = f_s.reshape(B * C, L)

#     corr_orig = get_pearson_correlation(f_t_reshaped, f_s_reshaped)
#
#     mean_corr_orig = corr_orig.mean()

#     t_low_reshaped = t_low_feat.reshape(B * C, L)
#     s_low_reshaped = s_low_feat.reshape(B * C, L)
#
#     corr_low = get_pearson_correlation(t_low_reshaped, s_low_reshaped)
#     mean_corr_low = corr_low.mean()
#
#     print(f"   low fre: {mean_corr_low.item():.4f}")
#
#     t_high_reshaped = t_high_feat.reshape(B * C, L)
#     s_high_reshaped = s_high_feat.reshape(B * C, L)
#

#     corr_high = get_pearson_correlation(t_high_reshaped, s_high_reshaped)
#     mean_corr_high = corr_high.mean()

#     print(f"   high-fre: {mean_corr_high.item():.4f}")






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




def create_imagenet_loaders(data_root, batch_size=128, num_workers=8, input_size=224):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dir = os.path.join(data_root, 'val')

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
    return None, val_loader  


def get_decouple_feature(x, mask_ratio=0.1):
    B, C, H, W = x.shape

    spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
    magnitude = spectrum.abs()

    magnitude_mean = torch.mean(magnitude, dim=1, keepdim=True)
    magnitude_flat = magnitude_mean.view(B, -1)
    threshold = torch.quantile(magnitude_flat, q=1 - mask_ratio, dim=-1, keepdim=True)
    threshold = threshold.unsqueeze(-1).unsqueeze(-1)

    mask = (magnitude_mean >= threshold)

    recon_low = torch.where(mask, spectrum, torch.tensor(0., device=x.device))
    low_freq_x = torch.fft.irfft2(recon_low, s=(H, W), dim=(-2, -1), norm='ortho')

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



if __name__ == '__main__':
    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    teacher_model = imagenet_model_dict['ResNet34'](num_classes=1000)
    pretrained_teacher = torch.load('Pretrain/ImageNet/ResNet34_vanilla/ckpt_resnet34.pth')
    teacher_model.load_state_dict(pretrained_teacher)
    teacher_model.eval().cuda()

    _, val_loader = create_imagenet_loaders(data_root='/data1/ImageNet2012', batch_size=512, num_workers=4)

    images_origin, _ = next(iter(val_loader))
    images_origin = images_origin.cuda()

    print(f"    Input Image Shape: {images_origin.shape}")

    

    # rotation
    angle = 9
    print(f"    ratation ratio: {angle} degrees")

    images_noisy = TF.rotate(images_origin, angle=angle)

    ############## noise
    # noise_sigma = 0.1
    # print(f"    gaussis noise: Sigma = {noise_sigma}")
    #
    # noise = torch.randn_like(images_origin, device=images_origin.device) * noise_sigma
    # images_noisy = images_origin + noise

    # scale 
    # scale_ratio = 0.1
    # print(f"    scale Ratio = {scale_ratio}")
    #
    # H, W = images_origin.shape[2], images_origin.shape[3]
    #
    # images_down = F.interpolate(
    #     images_origin,
    #     scale_factor=scale_ratio,
    #     mode='bilinear',
    #     align_corners=False
    # )
    #
    # images_noisy = F.interpolate(
    #     images_down,
    #     size=(H, W),
    #     mode='bilinear',
    #     align_corners=False
    # )


    with torch.no_grad():
        _, t_feats_origin = teacher_model(images_origin)
        _, t_feats_shifted = teacher_model(images_noisy)

    f_origin = t_feats_origin['feats'][4]  # [B, C, 7, 7]
    f_shifted = t_feats_shifted['feats'][4]

    B, C, H, W = f_origin.shape
    L = H * W
    print(f"    Feature Map Shape: {f_origin.shape}")

 
    RATIO = 0.1
    low_origin, high_origin = get_decouple_feature(f_origin, mask_ratio=RATIO)
    low_shifted, high_shifted = get_decouple_feature(f_shifted, mask_ratio=RATIO)
    # low_origin, high_origin = get_fixed_radiurs_feature(f_origin, radius=1.0)
    # low_shifted, high_shifted = get_fixed_radiurs_feature(f_shifted, radius=1.0)

    # t_mask, t_low_feat, t_high_feat = get_decouple_feature(f_t, mask_ratio=[0.10])
    # # s_mask, s_low_feat, s_high_feat = get_decouple_feature(f_s, mask_ratio=[0.1])
    # s_low_feat, s_high_feat = get_student_decouple_feat(f_s, t_mask)

    # -------------------------------------------------

    sim_orig = F.cosine_similarity(f_origin.reshape(B * C, L), f_shifted.reshape(B * C, L), dim=1).mean()


    sim_low = F.cosine_similarity(low_origin.reshape(B * C, L), low_shifted.reshape(B * C, L), dim=1).mean()

    sim_high = F.cosine_similarity(high_origin.reshape(B * C, L), high_shifted.reshape(B * C, L), dim=1).mean()

    print("\n" + "=" * 50)
    print("   Resluts analysis: Teacher vs Teacher (Shifted)")
    print("=" * 50)
    print(f" perturbation {angle} ")
    print("-" * 30)
    print(f"1. Original:  {sim_orig:.4f}")
    print(f"2. Low-Freq:  {sim_low:.4f} ")
    print(f"3. High-Freq: {sim_high:.4f} ")


    print("-" * 30)
    drop = sim_low - sim_high
    print(f"diff (Low - High): {drop:.4f}")


