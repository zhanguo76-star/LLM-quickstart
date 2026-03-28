from transformers import AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training
import torch

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v2', load_in_8bit=True, device_map='auto')
model = prepare_model_for_int8_training(model)
print("proj_out weight dtype:", model.proj_out.weight.dtype)
print("model.model.decoder.layer_norm weight dtype:", model.model.decoder.layer_norm.weight.dtype)
