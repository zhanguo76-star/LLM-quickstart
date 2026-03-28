import os
import torch
from datasets import DatasetDict, Dataset
from transformers import (
    AutoFeatureExtractor, AutoTokenizer, AutoProcessor, 
    AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

model_name_or_path = "openai/whisper-large-v2"
language = "Chinese (China)"
task = "transcribe"

# 1. Load feature extractor, tokenizer, processor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# 2. Mock dataset
dummy_audio = {"array": [0.0]*16000, "sampling_rate": 16000}
dummy_data = {
    "sentence": ["测试"] * 4,
    "audio": [dummy_audio] * 4
}
ds = Dataset.from_dict(dummy_data)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare_dataset)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 3. Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)

config = LoraConfig(r=4, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
peft_model = get_peft_model(model, config)

# FIX: Cast all float32 params to float16
for param in peft_model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

peft_model.config.use_cache = False

training_args = Seq2SeqTrainingArguments(
    output_dir="./tmp_test",
    per_device_train_batch_size=2,
    learning_rate=1e-3,
    max_steps=2,
    fp16=True,
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=ds,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()
print("Training finished successfully!")
