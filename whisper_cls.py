#%%
import os
import random
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore') 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#%%
CFG = {
    'SR' : 16000,
    'SEED' : 41,
    'ACUM' : 8,
    'BATCH_SIZE' : 32 ,
    'EPOCHS' : 20,
    'LR' : 1e-4
}
MODEL_NAME = "openai/whisper-base"
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
df.path = df.path.str.replace('./','./speech_data/')
train_df, val_df, _, _ = train_test_split(df,df['label'],test_size = 0.2, random_state = CFG['SEED'])
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

#%%
class CustomDataSet(Dataset):
    def __init__(self, file_list, labels, processor):
        self.file_list = file_list
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.file_list[idx])
        input = self.processor(audio[0], sampling_rate=CFG['SR'], return_tensors="pt").input_features[0]
        if self.labels is not None:
            return input, self.labels[idx]
        else:
            return input


# %%
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

train_dataset = CustomDataSet(train_df['path'], train_df['label'], processor)
valid_dataset = CustomDataSet(val_df['path'], val_df['label'], processor)

train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE']//CFG['ACUM'], shuffle = True, num_workers= 0)
valid_loader = DataLoader(valid_dataset, batch_size = CFG['BATCH_SIZE']//CFG['ACUM'], shuffle = False, num_workers= 0)
#%%

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = WhisperForAudioClassification.from_pretrained(MODEL_NAME)
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, 6)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output
# %%

def validation(model, valid_loader, criterion, device):
    model.eval()
    val_loss = []

    true_labels = []
    preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(iter(valid_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)#.logits
            loss = criterion(output, labels)

            val_loss.append(loss.item())

            preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

    accuracy = accuracy_score(true_labels, preds)
    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy

def train(model, train_loader, valid_loader, optimizer, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0.
    best_model = None
    step = 0
    running_loss = 0
    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = []
        model.train()
        for inputs, labels in tqdm(iter(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            output = model(inputs)#.logits
            loss = criterion(output, labels)

            (loss/CFG['ACUM']).backward()
            running_loss += loss.item()
            step += 1

            if step % CFG['ACUM']:
                continue

            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(running_loss / CFG['ACUM'])
            running_loss = 0

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, criterion, device)
        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
        
        if scheduler is not None:
            scheduler.step(valid_acc)

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model.state_dict(), f'ckp/best_model_score_{round(valid_acc, ndigits = 3)}.pt')
            print(f'{valid_acc}_model_save !')

        
# %%
#model = WhisperForAudioClassification.from_pretrained(MODEL_NAME,num_labels = 6)
model = BaseModel()
# for param in model.model.wav2vec2.feature_extractor.parameters():
#     param.requires_grad = False
#%%
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold_mode='abs', verbose=True)

train(model, train_loader, valid_loader, optimizer, scheduler, device)

#%%
infer_model = WhisperForAudioClassification.from_pretrained(MODEL_NAME,num_labels = 6)
infer_model.load_state_dict(torch.load('./ckp/best_model_score_0.579.pt'))
infer_model.eval()
infer_model = infer_model.to(device)
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

            output = model(x)#.logits

            preds += output.argmax(-1).detach().cpu().numpy().tolist()

    return preds
# %%
preds = inference(model,test_loader)
# %%
submission = pd.read_csv('./speech_data/sample_submission.csv')
submission['label'] = preds
submission.to_csv('./baseline_submission.csv', index=False)
# %%
np.unique(df.label,return_counts= True)
# %%
