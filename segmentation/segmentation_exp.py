import math
import os
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import cachestore
import cv2
import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_storage import AWSStorage
from PIL import Image
from pydensecrf.utils import create_pairwise_gaussian, unary_from_labels
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

parser = ArgumentParser()
# parser.add_argument('--metrics-path', type=Path, required=True)
parser.add_argument("--disable-cache", action="store_true", help="Disable cache")
parser.add_argument(
    "--cache-root", type=Path, default=".cachestore", help="Cache directory"
)
parser.add_argument(
    "--exp-name", type=str, required=True, help="Specifies a unique experiment name"
)
parser.add_argument(
    "--data-dir", type=Path, help="Directory with the data", default="./inputs"
)
args, _ = parser.parse_known_args()
print(time.ctime(), "Starting...")

epochs = 3

cache = cachestore.Cache(
    f"segmentation_{args.exp_name}_cache",
    disable=args.disable_cache,
    storage=AWSStorage(args.cache_root),
)

print(f"{cache.name=}")
print(f"{cache.settings=}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type.find("cpu") == -1:
    print(f"Running on {device} device")
else:
    print(Warning("Running on CPU instead of GPU!!!"))

IMAGE_PATH = f"{args.data_dir}/original_images/"
MASK_PATH = f"{args.data_dir}/label_images_semantic/"
EXP_PATH = args.cache_root / args.exp_name
EXP_PATH.mkdir(parents=True, exist_ok=True)


n_classes = 23


def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split(".")[0])

    return pd.DataFrame({"id": name}, index=np.arange(0, len(name)))


df = create_df()

# Reduce the number of datapoints to 20%
df = df.sample(frac=0.2, replace=False)

# split data
X_trainval, X_test = train_test_split(df["id"].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)


# Tuning the mean and std values
class DroneDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        self.cache = {}

        # Build the cache
        print("Transforming data")
        for idx in tqdm(range(len(self))):
            self.apply_transforms(idx)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # if self.cache.get(idx):
        return self.cache[idx]

    def apply_transforms(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + ".png", cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        self.cache[idx] = img, mask
        return img, mask

    def tiles(self, img, mask):
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches


params = {"width": 704, "height": 1056, "grid_distortion_p": 0.2}
t_train = A.Compose(
    [
        A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        A.GaussNoise(),
    ]
)

# t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
#                    A.GridDistortion(p=0.2)])
t_val = A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST)

# datasets


@cache(ignore={"epochs"})
def get_dataset(epochs, mean, std):
    """
    epoch: part of cache key. No other use in this function
    """
    train_sets = [
        DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
        for _ in range(epochs)
    ]
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)
    return train_sets, val_set


def pixel_accuracy(output, mask):
    with torch.no_grad():
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = (
                    torch.logical_and(true_class, true_label).sum().float().item()
                )
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


@cache(ignore={"patch", "epochs", "datasets"})
def train(
    stage1_params, epochs, datasets, max_lr, weight_decay, batch_size, patch=False
):
    model = smp.Unet(
        "mobilenet_v2",
        encoder_weights="imagenet",
        classes=23,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    model.to(device)

    lrs = []

    steps_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    running_loss = 0
    iou_score = 0
    accuracy = 0
    # training loop
    model.train()
    for epoch in range(epochs):
        train_loader = DataLoader(datasets[epoch], batch_size=batch_size, shuffle=True)
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            # forward
            output = model(image)
            loss = criterion(output, mask)

            # evaluation metrics
            output = torch.argmax(output, dim=1)
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()
    name = f"{EXP_PATH / uuid.uuid4().hex}.pt"
    torch.save(model, name)
    return name


def predict_image_mask_miou(model, image, mask, **crf_kwargs):
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)

        mask = mask.unsqueeze(0)
        output = model(image)
        output = F.softmax(output, dim=1)
        pred_mask = torch.argmax(output, dim=1).squeeze(0)

        score = mIoU(pred_mask, mask)

        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_mask_crf = crf(image, pred_mask.cpu().numpy(), **crf_kwargs)

        score_crf = mIoU(pred_mask_crf, mask.cpu())

    return output, score, score_crf


def crf(image, predicted_mask, **kwargs):
    gt_prob, sdims, compat = (
        kwargs.pop("gt_prob"),
        kwargs.pop("sdims"),
        kwargs.pop("compat"),
    )

    # Define CRF parameters
    num_classes = np.max(predicted_mask) + 1
    n_iterations = 5  # Number of CRF iterations

    # Convert the predicted mask to a unary potential
    predicted_mask = predicted_mask  # .astype(np.uint8)
    predicted_mask = cv2.resize(
        predicted_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    predicted_unary = unary_from_labels(
        predicted_mask, num_classes, gt_prob=gt_prob, zero_unsure=False
    )
    # Create a CRF object
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], num_classes)

    # Set the unary potential
    d.setUnaryEnergy(predicted_unary)

    # Create pairwise potentials (gaussian and bilateral)
    pairwise_energy = create_pairwise_gaussian(sdims=sdims, shape=image.shape[:2])

    # Add the pairwise energy to the CRF
    d.addPairwiseEnergy(pairwise_energy, compat=compat)

    # Inference (CRF optimization)
    masked_crf = np.argmax(d.inference(n_iterations), axis=0).reshape(image.shape[:2])
    # The 'masked_crf' now contains the refined segmentation mask after CRF post-processing

    # You can visualize the results or use them for further analysis as needed

    masked_crf = torch.tensor(masked_crf)

    return masked_crf


def miou_score(model, test_set, **crf_kwargs):
    score_iou = []
    score_iou_crf = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]

        output, score, score_crf = predict_image_mask_miou(
            model, img, mask, **crf_kwargs
        )
        score_iou.append(score)
        score_iou_crf.append(score_crf)

    print(f"Mean score_iou: {np.mean(score_iou)}±{np.std(score_iou)}")
    print(f"Mean score_iou_crf: {np.mean(score_iou_crf)}±{np.std(score_iou_crf)}")
    return np.mean(score_iou), np.mean(score_iou_crf)


def fit(epochs, max_lr, weight_decay, batch_size, mean, std, patch=False, **crf_kwargs):
    torch.cuda.empty_cache()

    fit_time = time.time()

    # Stage I: Data pre-processing and transformations
    stage1_params = [epochs, mean, std]
    train_sets, val_set = get_dataset(*stage1_params)

    end_1_time = time.time()

    # Stage II: Train the model
    name = train(
        stage1_params, epochs, train_sets, max_lr, weight_decay, batch_size, patch=patch
    )

    end_2_time = time.time()

    # Stage III: Running Validation and Smoothing the predicted mask
    model = torch.load(name)
    score_miou, score_miou_crf = miou_score(model, val_set, **crf_kwargs)
    end_3_time = time.time()

    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))

    cost_per_stage = [
        end_1_time - fit_time,
        end_2_time - end_1_time,
        end_3_time - end_2_time,
    ]
    return score_miou, score_miou_crf, cost_per_stage


def main(new_hp_dict):
    """
    new_hp_dict looks like:
        {
            "0__hp1": 0.34,
            "0__hp2": 0.34, ...,
            "2__hpn": 0.34
        }
    """

    hparams_stage_grouped: dict[dict] = {}
    for key in new_hp_dict:
        stage_id, hp_name = key.split("__")
        stage_id = int(stage_id)
        if hparams_stage_grouped.get(stage_id) is None:
            hparams_stage_grouped[stage_id] = {hp_name: new_hp_dict[key]}
        else:
            hparams_stage_grouped[stage_id].update({hp_name: new_hp_dict[key]})

    mean = {
        key: val
        for key, val in hparams_stage_grouped[0].items()
        if key.startswith("mean")
    }  # To be tuned
    std = {
        key: val
        for key, val in hparams_stage_grouped[0].items()
        if key.startswith("std")
    }

    mean = list(dict(sorted(mean.items())).values())
    std = list(dict(sorted(std.items())).values())

    # Stage II
    batch_size = hparams_stage_grouped[1]["batch_size"]
    max_lr = hparams_stage_grouped[1]["max_lr"]
    weight_decay = hparams_stage_grouped[1]["weight_decay"]

    # Stage III
    sdim = hparams_stage_grouped[2]["sdim"]
    sdims = (sdim, sdim)
    compat = hparams_stage_grouped[2]["compat"]
    gt_prob = hparams_stage_grouped[2]["gt_prob"]
    score_miou, score_miou_crf, cost_per_stage = fit(
        epochs,
        max_lr,
        weight_decay,
        batch_size,
        mean,
        std,
        patch=False,
        sdims=sdims,
        compat=compat,
        gt_prob=gt_prob,
    )

    return score_miou, score_miou_crf, cost_per_stage


# if __name__=="__main__":
#     main()
