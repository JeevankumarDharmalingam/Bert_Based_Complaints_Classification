import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,get_linear_schedule_with_warmup,AdamW,AutoTokenizer,AutoModel
import tez
from torch.utils.data import Dataset, DataLoader
from tez.datasets import GenericDataset
from tez import Model
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn import metrics,model_selection,preprocessing
import torchvision
import os
import sys
from tez.callbacks import EarlyStopping




class IMDB_Dataset(Dataset):
    def __init__(self, Text, Target, tokenizer, max_len=512):
        self.text = Text
        self.target = Target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        review = str(self.text[item])
        review = ' '.join(review.split())
        inputs = self.tokenizer.encode_plus(review, None, add_special_tokens=True, max_length=self.max_len,
                                            padding="max_length", return_token_type_ids=True,
                                            return_tensors="pt", truncation=True)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return {

            "input_ids": input_ids.squeeze(),
            "token_type_ids": token_type_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "targets": torch.tensor(self.target[item], dtype=torch.long)
        }


class Imdb_Model(Model):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 6)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = torch.argmax(targets, dim=1).cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        model = self.base_model
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        opt = optimizer
        return opt

    def fetch_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_loader)
        )
        return scheduler

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, targets=None):
        _, o_2 = self.base_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        b_o = self.dropout(o_2)
        output = self.out(b_o)

        # calculate loss here
        loss = nn.CrossEntropyLoss()(output, torch.max(targets, 1)[1])

        # calculate the metric dictionary here
        metric_dict = self.monitor_metrics(output, targets)
        return output, loss, metric_dict


if __name__ == '__main__':
    df = pd.read_csv('https://github.com/srivatsan88/YouTubeLI/blob/master/dataset/consumer_compliants.zip?raw=true',
                     compression='zip', sep=',', quotechar='"')

    target = df['Product'].unique()
    target = pd.get_dummies(df["Product"])
    Y = target
    X = df["Consumer complaint narrative"]

    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.15, stratify=Y.values,
                                                        random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataset = IMDB_Dataset(X_train,Y_train,tokenizer=tokenizer)
    valid_dataset = IMDB_Dataset(X_test,Y_test,tokenizer=tokenizer)
    data = train_dataset[5]
    print(tokenizer.decode(data["input_ids"]))
    model = Imdb_Model()

    es = EarlyStopping(monitor="valid_loss",mode= "min" ,model_path="model.bin",patience=3)

    model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_bs=16,
        valid_bs=8,
        device="cuda",
        epochs=5,
        callbacks= [es],
        fp16=True

    )
    model.save("model.bin")