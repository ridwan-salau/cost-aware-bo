

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import pickle
import json
from pathlib import Path
from argparse import ArgumentParser

from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler  # https://github.com/ildoonet/pytorch-gradual-warmup-lr
import albumentations as A
import geffnet

import cachestore

parser = ArgumentParser()

def number(val:str):
    try:
        return int(val)
    except ValueError:
        return float(val)
parser.add_argument('--metrics-path', type=Path, required=True)
parser.add_argument('--hps', nargs='+', type=str, required=True)
parser.add_argument("--disable-cache", action="store_true", help="Disable cache")

args = parser.parse_args()

print(time.ctime(), "Starting...")

device = torch.device('cuda')

cache = cachestore.Cache("melanoma_cache", disable=args.disable_cache)

# # Config


kernel_type = 'effnetb3_256_meta_9c_ext_5epo'
image_size = 256
use_amp = False
data_dir = './input/jpeg-melanoma-256x256'
data_dir2 = './input/jpeg-isic2019-256x256'
enet_type = 'efficientnet-b3'
batch_size = 64
num_workers = 4
init_lr = 3e-5
out_dim = 9
DEBUG = True

freeze_epo = 0
warmup_epo = 1
cosine_epo = 4
n_epochs = freeze_epo + warmup_epo + cosine_epo

use_external = '_ext' in kernel_type
use_meta = 'meta' in kernel_type

# # Read CSV & Target Preprocess


# df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
# df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, 'test', f'{x}.jpg'))


# df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
# df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
# # df_train['fold'] = df_train['tfrecord'] % 5
# tfrecord2fold = {
#     2:0, 4:0, 5:0,
#     1:1, 10:1, 13:1,
#     0:2, 9:2, 12:2,
#     3:3, 8:3, 11:3,
#     6:4, 7:4, 14:4,
# }
# df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
# df_train['is_ext'] = 0
# df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))


# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
# df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))


# if use_external:
#     df_train2 = pd.read_csv(os.path.join(data_dir2, 'train.csv'))
#     df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
#     df_train2['fold'] = df_train2['tfrecord'] % 5
#     df_train2['is_ext'] = 1
#     df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir2, 'train', f'{x}.jpg'))

#     df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
#     df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
#     df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

# diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
# df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
# mel_idx = diagnosis2idx['melanoma']
# diagnosis2idx


# # Preprocess Meta Data

if use_meta:
    # One-hot encoding of anatom_site_general_challenge feature
    # concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    # dummies = pd.get_dummies(df_train['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
    # df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    # # df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    # df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    # # df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    # df_train['sex'] = df_train['sex'].fillna(-1)
    # # df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    # df_train['age_approx'] /= 90
    # df_test['age_approx'] /= 90
    # df_train['age_approx'] = df_train['age_approx'].fillna(0)
    # # df_test['age_approx'] = df_test['age_approx'].fillna(0)
    # df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    # df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    # # # df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    # df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    # df_train['n_images'] = np.log1p(df_train['n_images'].values)
    # # df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    # train_images = df_train['filepath'].values
    # train_sizes = np.zeros(train_images.shape[0])
    # for i, img_path in enumerate(tqdm(train_images)):
    #     train_sizes[i] = os.path.getsize(img_path)
    # df_train['image_size'] = np.log(train_sizes)
    # test_images = df_test['filepath'].values
    # test_sizes = np.zeros(test_images.shape[0])
    # for i, img_path in enumerate(tqdm(test_images)):
    #     test_sizes[i] = os.path.getsize(img_path)
    # df_test['image_size'] = np.log(test_sizes)
    # df_train.to_csv("../input/preprocessed_train.csv", index=False)
    df_train = pd.read_csv("../input/preprocessed_train.csv")
    df_train["filepath"] = df_train.filepath.str.replace("../", "./")
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    mel_idx = diagnosis2idx['melanoma']

else:
    n_meta_features = 0

print(time.ctime(), "Done pre-processing metadata")

# # Define Dataset

class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None, pre_proc_path="../input/preprocessed"):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.pre_proc_path = pre_proc_path
        self.pre_proc_files = []

        if transform is not None:
            self.csv.set_index("image_name", inplace=True)
            import uuid
            os.makedirs(pre_proc_path, exist_ok=True)
            for ind, row in self.csv.iterrows():
                image = cv2.imread(row.filepath)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for i in range(n_epochs):
                    res = transform(image=image)
                    image_name = f"{os.path.basename(row.filepath)}{uuid.uuid4().hex}.jpg"
                    # print(res["image"].sum())
                    with open(os.path.join(pre_proc_path, image_name), "wb") as f:
                        pickle.dump(res["image"], f)
                    # cv2.imwrite(os.path.join(pre_proc_path, image_name), res["image"])
                    self.pre_proc_files.append(image_name)
                    
                    if not mode=="train":
                        break

    def __len__(self):
        if self.transform:
            return len(self.pre_proc_files)
        return self.csv.shape[0]

    def __getitem__(self, index):        
        if self.transform is not None:
            with open(os.path.join(self.pre_proc_path, self.pre_proc_files[index]), "rb") as f:
                image = pickle.load(f)
            index = self.pre_proc_files[index].split(".")[0]
        else:
            row = self.csv.iloc[index]
            image = cv2.imread(row.filepath)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(image.sum(), end="<<>>")
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        # print(image.sum())

        if use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.loc[index][meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.loc[index].target).long()

# # Augmentations
@cache()
def get_train_dataset(i_fold, bright_limit=0.2, cont_limit=0.2, p=0.5):
    transforms_train = A.Compose([
        A.Transpose(p=p),
        A.VerticalFlip(p=p),
        A.HorizontalFlip(p=p),
        A.RandomBrightness(limit=bright_limit, p=p),
        A.RandomContrast(limit=cont_limit, p=p),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=p),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=p),

        A.CLAHE(clip_limit=4.0, p=p),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=p),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=p),    
        A.Normalize()
    ])
    
    print(time.ctime(), "Transforming train data; fold:", fold)
    df_this = df_train[df_train['fold'] != i_fold].sample(batch_size * 12, random_state=0)
    dataset_train = SIIMISICDataset(df_this,  'train', 'train', transform=transforms_train)
    print(time.ctime(), "Done transforming train data")
    return dataset_train
    
    

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize()
])

# # Model


sigmoid = torch.nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
swish_layer = Swish_module()

class enetv2(nn.Module):
    def __init__(self, out_dim, n_meta_features=0, h1=512, h2=128, p=0.3):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=True)
        self.dropout = nn.Dropout(0.5)
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, h1),
                nn.BatchNorm1d(h1),
                Swish_module(),
                nn.Dropout(p=p),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                Swish_module(),
            )
            in_ch += h2
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            x = self.myfc(self.dropout(x))
        return x

# # Loss Function


criterion = nn.CrossEntropyLoss()


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

    
def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in (loader):
            
            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS==mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext==0]==mel_idx).astype(float), PROBS[is_ext==0, mel_idx])
        return val_loss, acc, auc, auc_20


# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def run(fold, hparams):
    print(time.ctime(), "Running fold", fold)
    stage1_hps = hparams[:3]
    stage2_hps = hparams[3:]
    
    i_fold = fold

    df_valid = df_train[df_train['fold'] == i_fold].sample(batch_size * 10, random_state=0)
    start = time.time()
    dataset_train = get_train_dataset(fold, *stage1_hps)
    end = time.time()
    with (EXP_DIR/"stage1_cost.json").open("w") as f:
        json.dump({"COST": end-start}, f)
    dataset_valid = SIIMISICDataset(df_valid, 'train', 'val', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers)

    @cache()
    def train_loop(fold, stage1_hps, init_lr, warmup_epo, h1, h2, drp_out_p):
        cosine_epo = n_epochs - warmup_epo
        model = enetv2(n_meta_features=n_meta_features, out_dim=out_dim, h1=h1, h2=h2, p=drp_out_p)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

        print(len(dataset_train), len(dataset_valid))
        
        total_iter = len(dataset_train)//batch_size
        train_loss = []
        
        print(time.ctime(), "Training loop starting...")
        for it, (data, target) in enumerate(train_loader):
            epoch = it // (total_iter//(n_epochs)) + 1
            if it % (total_iter//n_epochs) == 0:
                print(time.ctime(), 'Epoch:', epoch, "Iteration:", it)
                scheduler_warmup.step(epoch-1)

            model.train()
            
    #         for i, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = model(data, meta)
            else:
                data, target = data.to(device), target.to(device)
                logits = model(data)
            loss = criterion(logits, target)

            loss.backward()
            
            optimizer.step()
            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            # bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
            
        model.eval()
        val_loss, acc, auc, auc_20 = val_epoch(model, valid_loader, is_ext=df_valid['is_ext'].values)
        print(time.ctime(), "Fold:", fold, f"Val loss: {val_loss}; Acc: {acc}; Auc: {auc}; Auc_20: {auc_20}")
        torch.save(model.state_dict(), os.path.join(f'{kernel_type}_model_fold{i_fold}.pth'))        
        return auc
        
    start=time.time()        
    auc = train_loop(fold, stage1_hps, *stage2_hps)
    duration = time.time() - start
    return auc, duration
# # Run 5-Fold Training



if __name__=="__main__":
    EXP_DIR: Path = args.metrics_path
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print(__file__, 495, EXP_DIR)
    aucs = []
    durations = []
    for fold in range(5):
        # hparams = list(dict(bright_limit=0.2, cont_limit=0.2, p=0.5, init_lr=3e-5, warmup_epo=1, h1=512, h2=128, drp_out_p=0.5).values())
        hparams = [number(i) for i in args.hps]
        
        auc, duration = run(fold, hparams)
        aucs.append(auc)
        durations.append(duration)
    with (EXP_DIR/"obj.json").open("w") as f:
        json.dump({"OBJ": np.mean(auc)}, f)
    with (EXP_DIR/"stage2_cost.json").open("w") as f:
        json.dump({"COST": np.mean(duration)}, f)
