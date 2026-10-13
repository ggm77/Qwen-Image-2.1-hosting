import torch
import torch.nn as nn
import torch.nn.functional as F

class Int8Linear(nn.Module):
    """
    출력 채널별 대칭 int8 가중치 Linear.
    가중치는 int8로 저장하고 계산은 입력 dtype(bf16)으로 한다: y = (x @ q^T) * scale + bias

    생성자는 값이 없는 빈 껍데기(meta 텐서)를 만든다. 값은 from_linear()로 양자화해서 채우거나,
    저장해둔 state_dict를 load_state_dict(assign=True)로 불러와 채운다.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool, scale_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("q", torch.empty(out_features, in_features, dtype=torch.int8, device="meta"))
        self.register_buffer("scale", torch.empty(out_features, dtype=scale_dtype, device="meta"))
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=scale_dtype, device="meta"), requires_grad=False)
            if has_bias else None
        )

    @classmethod
    def from_linear(cls, lin: nn.Linear, scale_dtype: torch.dtype) -> "Int8Linear":
        module = cls(lin.in_features, lin.out_features, lin.bias is not None, scale_dtype)
        w = lin.weight.data.float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
        module.q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        module.scale = scale.squeeze(1).to(scale_dtype)
        if lin.bias is not None:
            module.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.q.to(x.dtype)) * self.scale.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)

def quantize_linears(module: nn.Module, scale_dtype: torch.dtype, empty: bool = False) -> int:
    """
    module 안의 모든 nn.Linear를 Int8Linear로 교체하고 교체한 개수를 반환한다.
    empty=True면 값을 계산하지 않고 빈 껍데기만 만든다 (가중치가 meta인 모델에 저장된 값을 불러올 때 사용).
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if empty:
                new = Int8Linear(child.in_features, child.out_features, child.bias is not None, scale_dtype)
            else:
                new = Int8Linear.from_linear(child, scale_dtype)
            setattr(module, name, new)
            count += 1
        else:
            count += quantize_linears(child, scale_dtype, empty)
    return count
