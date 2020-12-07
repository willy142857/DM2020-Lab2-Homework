import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, BertConfig


transformers.logging.set_verbosity_error()

class TweetDataset(Dataset):
    def __init__(self, texts, emotions, tokenizer, max_len):
        self.texts = texts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        emotion = self.emotions[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotions': torch.tensor(emotion, dtype=torch.long)
        }


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def train_model(model, dataloader: DataLoader, optimizer, device):
    model = model.train()

    losses = []
    acc_list = []
    with tqdm(dataloader) as t:
        t.set_description('Training')

        for data in t:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            emotions = data["emotions"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=emotions)
            loss = outputs.loss

            _, preds = torch.max(outputs.logits, dim=1)

            correct_pred_count = torch.sum(preds == emotions).double().item()
            acc = correct_pred_count / (emotions.shape[0])

            acc_list.append(acc)
            losses.append(loss.item())

            t.set_postfix(acc=np.mean(acc_list), loss=np.mean(losses))

            loss.backward()
            optimizer.step()

    return np.mean(acc_list), np.mean(losses)


def eval_model(model, dataloader: DataLoader, device):
    model = model.eval()

    losses = []
    acc_list = []
    with torch.no_grad():
        with tqdm(dataloader) as t:
            t.set_description('Validation')

            for data in t:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                emotions = data["emotions"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=emotions)
                loss = outputs.loss
                logits = outputs.logits

                _, preds = torch.max(logits, dim=1)
                correct_pred_count = torch.sum(preds == emotions).double().item()
                acc = correct_pred_count / (emotions.shape[0])
                acc_list.append(acc)
                losses.append(loss.item())

                t.set_postfix(acc=np.mean(acc_list), loss=np.mean(losses))

    return np.mean(acc_list), np.mean(losses)


def get_predictions(model, dataloader: DataLoader):
    model = model.eval()

    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds)

    predictions = torch.stack(predictions).cpu()

    return predictions


if __name__ == '__main__':
    dataset_dir = 'dm2020-hw2-nthu'
    dataset_path = Path(dataset_dir)

    train_df = pickle.load(open(dataset_path/'train.pkl', 'rb'))
    test_df = pickle.load(open(dataset_path/'test.pkl', 'rb'))

    PRE_TRAINED_MODEL_NAME = 'roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    x_train, x_val, y_train, y_val = train_test_split(
        train_df['text'].values, train_df['emotion'].values, test_size=0.2)

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    max_len = 256
    batch_size = 256
    num_workers = 4
    epochs = 10
    train_dataloader = DataLoader(TweetDataset(x_train, y_train, tokenizer=tokenizer, max_len=max_len),
                                batch_size=batch_size, num_workers=num_workers)

    val_dataloader = DataLoader(TweetDataset(x_val, y_val, tokenizer=tokenizer, max_len=max_len),
                                batch_size=batch_size, num_workers=num_workers)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BertConfig(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, return_dict=True, num_labels=8)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        train_acc, train_loss = train_model(model, train_dataloader, optimizer, device)

        val_acc, val_loss = eval_model(model, val_dataloader, device)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, return_dict=True, num_labels=8)
    model = model.to(device)
    model.load_state_dict(torch.load('best_model_state.bin'))

    test_dataloader = DataLoader(TestDataset(test_df['text'].values, tokenizer=tokenizer, max_len=max_len),
                                batch_size=batch_size, num_workers=num_workers)

    predictions = get_predictions(model, test_dataloader)

    ids = pd.DataFrame(test_df['tweet_id'].values, columns=['id'])
    emotions = pd.DataFrame([le.classes_[pred] for pred in predictions], columns=['emotion'])

    submission_df = pd.concat([ids, emotions], axis=1)
    submission_df.to_csv('submission.csv', index=False)
