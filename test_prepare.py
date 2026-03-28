from transformers import AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training
import torch

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v2', load_in_8bit=True, device_map='auto')
model = prepare_model_for_int8_training(model)
print("conv1 weight:", model.model.encoder.conv1.weight.dtype)
print("conv1 bias:", model.model.encoder.conv1.bias.dtype)
