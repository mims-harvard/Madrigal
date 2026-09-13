import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer


device = "cuda"


class LM(nn.Module):
    def __init__(self, decoder, lm_model):
        super(LM, self).__init__()
        self.decoder = decoder
        self.lm_model = lm_model

    def forward(self, input_ids):
        if self.lm_model == "mistralai/Mistral-7B-v0.1":
            transformer_outputs = self.decoder(
                input_ids["input_ids"], attention_mask=input_ids["attention_mask"]
            )
            hidden_states = transformer_outputs[0][:, 0, :]
        else:
            bert_embeddings = self.decoder.embeddings(input_ids=input_ids["input_ids"])
            extended_attention_mask = self.decoder.get_extended_attention_mask(
                input_ids["attention_mask"], input_ids["input_ids"].size()
            )
            outputs = self.decoder.encoder(
                bert_embeddings, attention_mask=extended_attention_mask
            )
            sequence_output = outputs[0]
            hidden_states = self.decoder.pooler(sequence_output)
        return hidden_states


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text


def collate_fn(batch, tokenizer):
    inputs = tokenizer(batch, padding="longest", return_tensors="pt")
    return inputs


def get_embeddings_labels_batched(lm_model, paraphrase, use_label, base_path):
    print("\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    if lm_model == "mistralai/Mistral-7B-v0.1":
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_length = len(tokenizer)

    print("\nLoading model")
    decoder = AutoModel.from_pretrained(lm_model).to(device)
    model = LM(decoder, lm_model).to(device)

    for split in ["train", "eval"]:
        print(f"\nOn split {split}")

        if not os.path.exists(os.path.join(base_path, split)):
            os.makedirs(os.path.join(base_path, split))

        df = pd.read_csv(
            os.path.join(base_path, f"paraphrased_dataset_new/{split}_df.csv")
        )
        
        for i in range(10):
            
            print(f'On description type {i}')
            
            descriptions = df[f"descriptions_{i}"].tolist()
            label_descriptions = np.unique(descriptions)

            print("Running tokenizer")

            dataset = TextDataset(label_descriptions.tolist())
            loader = DataLoader(
                dataset, batch_size=40, collate_fn=partial(collate_fn, tokenizer=tokenizer), shuffle=False
            )
            
            print("Running model")
            outputs = []
            for batch in tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    example_outputs = model(batch)
                    outputs.append(example_outputs)

            outputs = torch.cat(outputs, dim=0)

            assert outputs.size(dim=0) == len(label_descriptions)

            torch.save(
                outputs,
                os.path.join(base_path, "paraphrased_dataset_new", f"{split}_descriptions_{i}_embeddings.pt")
            )

    return os.path.join(base_path, "paraphrased_dataset_new")



def get_embeddings_labels(lm_model, paraphrase, use_label, base_path):
    print("\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    if lm_model == "mistralai/Mistral-7B-v0.1":
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_length = len(tokenizer)

    print("\nLoading model")
    decoder = AutoModel.from_pretrained(lm_model).to(device)
    model = LM(decoder, lm_model).to(device)

    for split in ["train", "eval"]:
        print(f"\nOn split {split}")

        if not os.path.exists(os.path.join(base_path, split)):
            os.makedirs(os.path.join(base_path, split))

        if paraphrase:
            df = pd.read_csv(
                os.path.join(base_path, f"augmented_dataset_2/{split}_df.csv")
            )
            descriptions = df["descriptions"].tolist()
            label_descriptions = np.unique(descriptions)
            
        elif use_label:
            df = pd.read_csv(
                os.path.join(base_path, f"label_dataset/{split}_df.csv")
            )
            descriptions = df["label_descriptions"].tolist()
            label_descriptions = np.unique(descriptions)
            
        else:
            df = pd.read_csv(os.path.join(base_path, f"dataset/{split}_df.csv"))
            descriptions = df["descriptions"].tolist()
            label_descriptions = np.unique(descriptions)

        print("Running tokenizer")
        example_encoding = tokenizer.batch_encode_plus(
            label_descriptions.tolist(), padding=True, return_tensors="pt"
        )

        example_encoding = example_encoding.to(device)

        print("Running model")
        with torch.no_grad():
            example_outputs = model(example_encoding)
            
        if paraphrase:
            torch.save(
                example_outputs,
                os.path.join(base_path, "augmented_dataset_2", f"{split}_label_embeddings.pt")
            )
            
        elif use_label:
            torch.save(
                example_outputs,
                os.path.join(base_path, "label_dataset", f"{split}_label_embeddings.pt")
            )
            
        else:
            torch.save(
                example_outputs,
                os.path.join(base_path, "dataset", f"{split}_label_embeddings.pt"),
            )

    return os.path.join(base_path, "dataset")


def get_embeddings_all_ddis(lm_model, paraphrase, batch_size, base_path):
    print("\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    if lm_model == "mistralai/Mistral-7B-v0.1":
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_length = len(tokenizer)

    print("\nLoading model")
    decoder = AutoModel.from_pretrained(lm_model).to(device)
    model = LM(decoder, lm_model).to(device)

    for split in ["train", "eval"]:

        print(f"\nOn split {split}")

        if not os.path.exists(os.path.join(base_path, split)):
            os.makedirs(os.path.join(base_path, split))

        if paraphrase:
            df = pd.read_csv(f"augmented_dataset_2/{split}_df.csv")
        else:
            df = pd.read_csv(f"dataset/{split}_df.csv")

        descriptions = df["drug_descriptions"].tolist()

        dataset = TextDataset(descriptions, tokenizer)
        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )

        for i, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.no_grad():
                example_outputs = model(batch)

            torch.save(
                example_outputs.cpu(),
                os.path.join(base_path, split, f"embeddings_{i//batch_size}.pt"),
            )

    return os.path.join(base_path, "train"), os.path.join(base_path, "eval")
