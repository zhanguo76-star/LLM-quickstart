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

# We ONLY cast inputs to Linear8bitLt to float16 using a forward_pre_hook.
# This ensures ctx.dtype_A is float16, avoiding the bitsandbytes backward dtype mismatch.
for module in peft_model.modules():
    if isinstance(module, bnb.nn.Linear8bitLt):
        module.register_forward_pre_hook(
            lambda m, args: (args[0].to(torch.float16), *args[1:])
        )

input_features = torch.randn(1, 80, 3000).cuda() # From dataloader, typically float32
decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()
labels = torch.tensor([[50258, 50259, 50359, 50363]]).cuda()

with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = peft_model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=labels)
    loss = outputs.loss

print("Loss dtype:", loss.dtype)
loss.backward()
print("Backward successful!")
