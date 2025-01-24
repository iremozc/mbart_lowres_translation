

import os
import json
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

train_file = "data/train.txt"
validation_file = "data/validation.txt"
test_file = "data/test.txt"

#Define Hyperparameters and Training Configurations
learning_rate = 3e-5
batch_size = 4
num_train_epochs = 10
fp16 = True

# Load the tokenizer
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
if "teo_UG" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["teo_UG"] = len(tokenizer.lang_code_to_id)
if "ach_UG" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["ach_UG"] = len(tokenizer.lang_code_to_id)

def load_and_tokenize_dataset(file_path, tokenizer):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line)  # Use eval because the dataset isn't strictly JSON
            target_text = data['eng']  # English (target language)
            tgt_lang = "en_XX"

	    # Check for Ateso or Acholi source text
            if "teo" in data:
                source_text_at = data['teo']  # Ateso source text
                src_lang = "teo_UG"            # Language code for Ateso
	        # Set tokenizer language codes
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang            
		# Tokenize source and target text
                source_ids = tokenizer(source_text_at, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]
                target_ids = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]

	        # Append Ateso-English pair
                dataset.append({
                    'input_ids': source_ids,
                    'labels': target_ids,
                    'source_text': source_text_at,
                    'target_text': target_text,
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang
                })
            if 'ach' in data:
                source_text_ach = data['ach']
                src_lang = "ach_UG"  # Using teo_UG for Acholi as well

                # Set tokenizer language codes
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang
                
                source_ids = tokenizer(source_text_ach, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]
                target_ids = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]

                # Append Ateso-English pair
                dataset.append({
                    'input_ids': source_ids,
                    'labels': target_ids,
                    'source_text': source_text_ach,
                    'target_text': target_text,
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang
                })

    return Dataset.from_list(dataset)


# Load the datasets with tokenization
train_dataset = load_and_tokenize_dataset(train_file, tokenizer)
validation_dataset = load_and_tokenize_dataset(validation_file, tokenizer)
test_dataset = load_and_tokenize_dataset(test_file, tokenizer)

# Step 4: Load the mBART Model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]  # Adjust this to match the target language

# Step 5: Define Data Collator
# DataCollatorForSeq2Seq will pad inputs dynamically to the longest sequence in the batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Step 6: Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./cross_learning_out_dir",
    eval_strategy="epoch",  # Updated based on the deprecation warning
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    fp16=fp16,  # Enable mixed precision training if supported
    report_to="none",  # Disable W&B integration
    run_name="mbart_finetuning_run",  
    load_best_model_at_end=True,  # Load the best model found during training
    metric_for_best_model="eval_loss",  # Use validation loss as the metric for determining the best model
    greater_is_better=False,  # Lower loss is better
    save_strategy="epoch",
    save_steps=500,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)
print("Training has started")
# Step 7: Create the Trainer Object
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the Model
trainer.train()
