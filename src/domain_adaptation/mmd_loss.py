import torch


def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5):
    """计算最大均值差异(MMD)损失

    Args:
        source_features: 源域特征 [batch_size, feature_dim]
        target_features: 目标域特征 [batch_size, feature_dim]
        kernel_mul: 核函数带宽乘数
        kernel_num: 核函数数量

    Returns:
        MMD损失值
    """
    batch_size = source_features.size(0)
    total_size = batch_size * 2

    features = torch.cat([source_features, target_features], dim=0)

    kernel_val = 0
    for sigma in range(-kernel_num, kernel_num + 1):
        sigma = kernel_mul ** sigma
        pairwise_dist = torch.cdist(features, features, p=2)
        kernel_val += torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))

    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd
