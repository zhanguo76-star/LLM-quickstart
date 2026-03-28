import torch
a = torch.randn(2).half()
b = torch.randn(2).float()
print("a dtype:", a.dtype)
print("b dtype:", b.dtype)
c = a.to(torch.float16).mul_(b)
print("c dtype:", c.dtype)
d = a.to(torch.float16).mul(b)
print("d dtype:", d.dtype)
