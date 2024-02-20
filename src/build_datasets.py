from datasets import load_dataset, DatasetDict

train_dataset = load_dataset(
    'json',
    data_files="stair_captions_v1.2_train_with_prompt.json",
    split="train",
)
val_dataset = load_dataset(
    'json',
    data_files="stair_captions_v1.2_val_with_prompt.json",
    split="train"
)
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})
dataset.save_to_disk("stair-captions-prompts")
dataset = load_dataset("stair-captions-prompts")
print(dataset)
# dataset.push_to_hub("your-repo/dataset-path", private=True)
