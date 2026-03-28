import torch
from transformers import AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v2', load_in_8bit=True, device_map='auto')
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)

config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)

# FIX: Cast all float32 params to float16
for param in peft_model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

# peft_model.base_model.model.proj_out = CastOutputToFloat(peft_model.base_model.model.proj_out)

input_features = torch.randn(1, 80, 3000).half().cuda()
decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()
labels = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()

with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = peft_model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=labels)
    loss = outputs.loss

print("Loss dtype:", loss.dtype)
print("Logits dtype:", outputs.logits.dtype)
loss.backward()
print("Backward successful!")
