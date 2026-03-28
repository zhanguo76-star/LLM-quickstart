import torch
import bitsandbytes as bnb
from transformers import AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v2', load_in_8bit=True, device_map='auto')
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)

config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)

# FIX 1: Cast all float32 params to float16 to save memory and keep things consistent
for param in peft_model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

# FIX 2: Add hooks to Linear8bitLt to force float16
for module in peft_model.modules():
    if isinstance(module, bnb.nn.Linear8bitLt):
        # Forward pre-hook: cast input to float16 so ctx.dtype_A is float16
        module.register_forward_pre_hook(
            lambda m, args: (args[0].to(torch.float16), *args[1:])
        )
        # Backward hook: cast grad_output to float16
        module.register_full_backward_hook(
            lambda m, grad_input, grad_output: tuple(g.to(torch.float16) if g is not None else None for g in grad_input)
        )

input_features = torch.randn(1, 80, 3000).half().cuda()
decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()
labels = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()

with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = peft_model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=labels)
    loss = outputs.loss

print("Loss dtype:", loss.dtype)
loss.backward()
print("Backward successful!")
