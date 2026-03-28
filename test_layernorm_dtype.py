import torch
ln = torch.nn.LayerNorm(10).half().cuda()
x = torch.randn(2, 10).cuda() # float32
y = ln(x)
print("Input dtype:", x.dtype)
print("Weight dtype:", ln.weight.dtype)
print("Output dtype:", y.dtype)
