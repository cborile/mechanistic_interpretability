import os
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.optim import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import seed_everything

def load_train_val_data(exp_name, dataset_name, topics_list, topic_col_name, input_data_path, random_state, save=True):
    if dataset_name == 'DAIGTV2':
        train_dataset = "train_v2_drcat_02.csv"
        data_input = os.path.join(input_data_path, "input", train_dataset)
        dataset_all = pd.read_csv(data_input)
    elif dataset_name == 'DAIGTV2lda':
        train_dataset = "train_v2_drcat_02_lda.csv"
        data_input = os.path.join(input_data_path, "input", train_dataset)
        dataset_all = pd.read_csv(data_input)
    elif dataset_name == 'HC3':
        train_dataset = "hc3_all.json"
        data_input = os.path.join(input_data_path, "input", train_dataset)
        dataset_all = pd.read_json(data_input, orient='records', lines=True)
        dataset_all = dataset_all[dataset_all.text.apply(len)>10].copy()
    elif dataset_name == 'polarity':
        train_dataset = "trainfakenews_ft.csv"
        data_input = os.path.join(input_data_path, "input", train_dataset)
        dataset_all = pd.read_csv(data_input)
    else:
        raise NotImplementedError
    
    if len(topics_list)>0:
        dataset = dataset_all[dataset_all[topic_col_name].isin(topics_list)].copy()
    else:
        dataset = dataset_all.copy()
    print(dataset['label'].value_counts())

    ## PREPARE TRAIN TEST SPLIT
    index_sample = dataset.groupby(['label', topic_col_name]).sample(1000, random_state=random_state).index.tolist()
    dataset.loc[:, 'training_set'] = 0
    dataset.loc[index_sample, 'training_set'] = 1
    print(dataset.training_set.value_counts())

    train_data = dataset[dataset['training_set'] == 1].copy()
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    if save:
        # save to json
        output_path = os.path.join(input_data_path, "input", f"data_train_{dataset_name}_{exp_name}_rs_{random_state}.json")
        train_data.to_json(
            output_path, 
            orient='records', 
            lines=True)

    return train_data, val_data

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# Function to create data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )

def train(data_loader, model, optimizer, scheduler, device, num_epochs):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        start_time_epoch = time.time()

        # Using tqdm for Jupyter Notebooks
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                model.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                #print(loss)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).cpu().numpy().mean()
                total_accuracy += accuracy

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Update tqdm bar
                tepoch.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}")

        end_time_epoch = time.time()
        epoch_duration = end_time_epoch - start_time_epoch
        avg_train_loss = total_loss / len(data_loader)
        avg_train_accuracy = total_accuracy / len(data_loader)

        # Print epoch-level summary
        print(f"End of Epoch {epoch + 1}/{num_epochs}, Duration: {epoch_duration:.2f}s, Average Loss: {avg_train_loss:.4f}, Average Accuracy: {avg_train_accuracy:.4f}")

    return total_loss


# Helper function for calculating accuracy
def calculate_accuracy(preds, labels):
    return np.sum(np.argmax(preds, axis=1).flatten() == labels.flatten()) / len(labels.flatten())


def test(model, data_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            total_accuracy += calculate_accuracy(logits, label_ids)

    return total_loss / len(data_loader), total_accuracy / len(data_loader)

def main():
    parser = argparse.ArgumentParser(description='Text detector (BERT-based)')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--topics_list', type=str, nargs='*', default=[])
    parser.add_argument('--topic_col_name', type=str, default="prompt_name") 
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--model_path', type=str, default="../models/")
    parser.add_argument('--dataset', type=str, default="DAIGTV2")
    parser.add_argument('--base_model', type=str, default="bert-base-cased")
    parser.add_argument('--rs', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--lr', type=int, default=2e-5)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    print(args)

    seed_everything(args.rs)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("device:", device)
    print("hf base model:", args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, clean_up_tokenization_spaces=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, output_hidden_states=False).to(device)

    train_data, val_data = load_train_val_data(
        args.exp_name, 
        args.dataset, 
        args.topics_list, 
        args.topic_col_name, 
        args.data_path, 
        args.rs)
    print(train_data.shape, val_data.shape)

    train_data_loader = create_data_loader(train_data, tokenizer, args.max_tokens, args.bs)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    total_loss = train(train_data_loader, model, optimizer, scheduler, device, args.num_epochs)
    print(total_loss)

    val_data_loader = create_data_loader(val_data, tokenizer, args.max_tokens, args.bs)
    # Evaluate after training
    val_loss, val_accuracy = test(model, val_data_loader, device)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Save trained model
    model_name = f"detector-{args.base_model}-{args.dataset}-{args.exp_name}"
    model.save_pretrained(os.path.join(args.model_path, model_name))

if __name__ == "__main__":
    main()


# Experiments:

# Namespace(exp_name='test', topics_list=['Car-free cities'], topic_col_name='prompt_name', device=0, data_path='../../data/', model_path='../models/', dataset='DAIGTV2', base_model='bert-base-cased', rs=0, num_epochs=4, lr=2e-05, bs=32, max_tokens=512)
# device: cuda:0
# hf base model: bert-base-cased
# Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# label
# 0    2666
# 1    2051
# Name: count, dtype: int64
# training_set
# 0    2717
# 1    2000
# Name: count, dtype: int64
# (1800, 6) (200, 6)
# /home/claudio/miniconda3/envs/mechinterp/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   warnings.warn(
# Epoch 1/4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57/57 [00:44<00:00,  1.27batch/s, accuracy=1.00, loss=0.000701]
# End of Epoch 1/4, Duration: 44.91s, Average Loss: 0.1679, Average Accuracy: 0.9337
# Epoch 2/4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57/57 [00:44<00:00,  1.28batch/s, accuracy=1.00, loss=0.000283]
# End of Epoch 2/4, Duration: 44.39s, Average Loss: 0.0113, Average Accuracy: 0.9973
# Epoch 3/4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57/57 [00:44<00:00,  1.28batch/s, accuracy=1.00, loss=0.00023]
# End of Epoch 3/4, Duration: 44.48s, Average Loss: 0.0075, Average Accuracy: 0.9978
# Epoch 4/4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57/57 [00:44<00:00,  1.29batch/s, accuracy=1.00, loss=0.000172]
# End of Epoch 4/4, Duration: 44.31s, Average Loss: 0.0038, Average Accuracy: 0.9995
# 0.21394332459021825
# Validation Loss: 0.0213
# Validation Accuracy: 0.9955
