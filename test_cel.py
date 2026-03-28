import torch
import torch.nn.functional as F
logits = torch.randn(2, 10).half().cuda()
labels = torch.tensor([1, 2]).cuda()
loss = F.cross_entropy(logits, labels)
print("Loss dtype:", loss.dtype)
loss.backward()
print("Grad dtype:", logits.grad.dtype)
