from transformers import AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training, get_peft_model, LoraConfig
import torch

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v2', load_in_8bit=True, device_map='auto')
model = prepare_model_for_int8_training(model)
config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)

print("Before casting:")
print("proj_out:", peft_model.base_model.model.proj_out.weight.dtype)
print("conv1:", peft_model.base_model.model.model.encoder.conv1.weight.dtype)
print("layer_norm:", peft_model.base_model.model.model.encoder.layer_norm.weight.dtype)

for name, param in peft_model.named_parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

print("\nAfter casting:")
print("proj_out:", peft_model.base_model.model.proj_out.weight.dtype)
print("conv1:", peft_model.base_model.model.model.encoder.conv1.weight.dtype)
print("layer_norm:", peft_model.base_model.model.model.encoder.layer_norm.weight.dtype)
