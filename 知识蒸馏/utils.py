
import torch.nn.functional as F
import torch


def compute_fkl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "mean",
    temp: float = 2.5,
    ) -> torch.Tensor:
        """
        优化版：在 F.kl_div 阶段直接处理 reduction 逻辑
        """
        # 温度调整
        temp = max(temp, 2e-6)
        logits = logits / temp
        teacher_logits = teacher_logits / temp
        
        # 数值稳定计算
        log_probs = F.log_softmax(logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Padding 掩码（提前计算）
        pad_mask = target.eq(padding_id)
        valid_elements = (~pad_mask).sum(dim=1)  # 每个样本的有效token数 [batch_size]
        
        # 根据 reduction 模式选择计算方式
        if reduction == "sum":
            kl_elements = F.kl_div(
                input=log_probs,
                target=teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1).masked_fill(pad_mask, 0.0).sum()  # 直接求和
            
        elif reduction == "mean":
            kl_elements = F.kl_div(
                input=log_probs,
                target=teacher_probs,
                reduction='none',
                log_target=False
            )
            # 对非padding位置求平均
            kl_sum = kl_elements.sum(dim=-1).masked_fill(pad_mask, 0.0).sum()
            non_pad_total = valid_elements.sum().clamp(min=1)  # 避免除零
            kl_elements = kl_sum / non_pad_total
            
        elif reduction == "batch_mean":
            kl_elements = F.kl_div(
                input=log_probs,
                target=teacher_probs,
                reduction='none',
                log_target=False
            )
            # 对每个样本求平均后再对batch平均
            kl_per_sample = kl_elements.sum(dim=-1).masked_fill(pad_mask, 0.0)  # [batch_size]
            kl_per_sample = kl_per_sample / valid_elements.clamp(min=1)  # 每个样本的均值
            kl_elements = kl_per_sample.mean()
            
        elif reduction == "none":
            kl_elements = F.kl_div(
                input=log_probs,
                target=teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1)  # 保持 [batch_size, seq_len]
        else:
            raise ValueError(f"不支持的 reduction 模式: {reduction}")
        
        return kl_elements