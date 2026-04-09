import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层(GRL)，用于DANN对抗训练

    在前向传播中恒等映射，在反向传播中反转梯度方向并乘以alpha系数。
    alpha通常随训练进度从0逐渐增大到1。
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None


def dann_loss(source_domain_output, target_domain_output, source_labels, alpha=1.0):
    """计算DANN域判别损失

    Args:
        source_domain_output: 源域的域判别器输出
        target_domain_output: 目标域的域判别器输出
        source_labels: 源域标签（未使用，保留接口一致性）
        alpha: GRL的alpha系数

    Returns:
        域对抗损失值
    """
    batch_size = source_domain_output.size(0)

    source_domain_labels = torch.zeros(batch_size, dtype=torch.long, device=source_domain_output.device)
    target_domain_labels = torch.ones(batch_size, dtype=torch.long, device=target_domain_output.device)

    domain_loss = F.cross_entropy(source_domain_output, source_domain_labels) + \
                  F.cross_entropy(target_domain_output, target_domain_labels)

    return domain_loss
