import torch
import torch.nn as nn
import torch.nn.functional as F

class Int8Linear(nn.Module):
    """
    출력 채널별 대칭 int8 가중치 Linear.
    가중치는 int8로 저장하고 계산은 입력 dtype(bf16)으로 한다: y = (x @ q^T) * scale + bias
    """

    def __init__(self, lin: nn.Linear, scale_dtype: torch.dtype):
        super().__init__()
        w = lin.weight.data.float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
        self.register_buffer("q", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).to(scale_dtype))
        self.bias = (
            None if lin.bias is None
            else nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.q.to(x.dtype)) * self.scale.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)

def quantize_linears(module: nn.Module, scale_dtype: torch.dtype) -> int:
    """module 안의 모든 nn.Linear를 Int8Linear로 교체하고 교체한 개수를 반환한다."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child, scale_dtype))
            count += 1
        else:
            count += quantize_linears(child, scale_dtype)
    return count
