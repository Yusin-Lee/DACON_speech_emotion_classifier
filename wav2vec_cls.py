#%%
import os
import random
import warnings
from sklearn.model_selection import train_test_split

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformers import AutoModel, Wav2Vec2FeatureExtractor,AutoModelForAudioClassification

import warnings
warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#%%
CFG = {
    'SR':16000,
    'SEED':42,
    'BATCH_SIZE':8, 
    'TOTAL_BATCH_SIZE':32, 
    'EPOCHS':30,
    'LR':1e-3,
}
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
#%%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
# %%
df = pd.read_csv('./speech_data/train.csv')
df = df[:1000]
df.path = df.path.str.replace('./','./speech_data/')
train_df, valid_df, _, _ = train_test_split(df,df['label'],test_size = 0.2, random_state = CFG['SEED'])
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
#%%
np.unique(df.label)
#%%
def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        feature.append(speech_array)
    return feature

train_x = speech_file_to_array_fn(train_df)
valid_x = speech_file_to_array_fn(valid_df)
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

#%%
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, processor):
        self.x = x
        self.y = y
        self.processor = processor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_values = self.processor(self.x[idx], sampling_rate=CFG['SR'], return_tensors="pt", padding=True).input_values
        if self.y is not None:
            return input_values.squeeze(), self.y[idx]
        else:
            return input_values.squeeze()
#%%
def collate_fn(batch):
    audio, label = zip(*batch)
    audio = pad_sequence([torch.tensor(xi) for xi in audio], batch_first=True)
    label = pad_sequence([torch.tensor([yi]) for yi in label], batch_first=True)
    return audio, label

def create_data_loader(dataset, batch_size, shuffle, collate_fn, num_workers=0):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=num_workers
                      )
# %%

train_dataset = CustomDataSet(train_x, train_df['label'], processor)
valid_dataset = CustomDataSet(valid_x, valid_df['label'], processor)

train_loader = create_data_loader(train_dataset, CFG['BATCH_SIZE'], True, collate_fn, 0)
valid_loader = create_data_loader(valid_dataset, CFG['BATCH_SIZE'], False, collate_fn, 0)

# %%
class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        self.model.freeze_base_model()
        self.model.classifier = nn.Linear(256, 6)
    def forward(self, x):
        output = self.model(x)
        return output
# %%
# model = BaseModel()
# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# input, vales = next(iter(train_loader))
# output = model(input)
# output
#%%
def validation(model, valid_loader, creterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iter(valid_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            output = model(x).logits
            loss = creterion(output, y)

            val_loss.append(loss.item())

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += predicted.eq(y).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy

def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
    model.to(device)
    creterion = nn.CrossEntropyLoss().to(device)

    best_model = None
    best_acc = 0

    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = []
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            optimizer.zero_grad()
            
            output = model(x).logits
            loss = creterion(output, y)
            
            loss.backward()

            if (i+1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, creterion)

        if scheduler is not None:
            scheduler.step(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
    
    print(f'best_acc:{best_acc:.5f}')

    return best_model

        
# %%
model = BaseModel()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
infer_model = train(model, train_loader, valid_loader, optimizer, scheduler)
# %%
import time

accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
model = BaseModel()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])

# for param in model.model.wav2vec2.feature_extractor.parameters():
#     param.requires_grad = False
model.to(device)
creterion = nn.CrossEntropyLoss().to(device)

to_device_time = 0
forward_time = 0
loss_time = 0
backward_time = 0


for epoch in range(1, CFG['EPOCHS']+1):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        start = time.time()
        x = x.to(device)
        y = y.flatten().to(device)
        to_device_time += start - time.time()

        optimizer.zero_grad()
        
        start = time.time()
        output = model(x)
        forward_time += start - time.time()
        
        start = time.time()
        loss = creterion(output, y)
        loss_time += start - time.time()

        start = time.time()
        (loss/accumulation_step).backward()
        backward_time += start - time.time()

        if (i+1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(loss)
#%%
output.size()
#%%

print(to_device_time)
print(forward_time)
print(loss_time)
print(backward_time)
for i in model.model.wav2vec2.encoder.parameters():
    print(i.requires_grad)
#%%
print(model)
# %%

test_df = pd.read_csv('./speech_data/test.csv')
test_df.path = test_df.path.str.replace('./','./speech_data/')
test_dataset = CustomDataSet(test_df['path'], None, processor)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE']//CFG['ACUM'], shuffle = False, num_workers= 0)
# %%
def inference(model, test_loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in tqdm(iter(test_loader)):
            x = x.to(device)

            output = model(x)

            preds += output.argmax(-1).detach().cpu().numpy().tolist()

    return preds
# %%
preds = inference(model,test_loader)
# %%
submission = pd.read_csv('./speech_data/sample_submission.csv')
submission['label'] = preds
submission.to_csv('./baseline_submission.csv', index=False)
# %%

