import torch
ln = torch.nn.LayerNorm(10).half().cuda()
x = torch.randn(2, 10).cuda() # float32
with torch.cuda.amp.autocast(dtype=torch.float16):
    y = ln(x)
print("Autocast with Half weights - Output dtype:", y.dtype)
