import os
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import monai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.model import PneuSam
from datasets.semantic_seg import PneumothoraxDataset, PneumoSampler

# set seeds
torch.manual_seed(49)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-dir_dataset",
    type=str,
    default="input/dataset1024",
    help="dir dataset",
)
parser.add_argument("-valEpoch", type=int, default=10)
parser.add_argument("-task_name", type=str, default="SAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="sam_ckpt/sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./experiment")
# train
parser.add_argument("-num_epochs", type=int, default=500)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=2)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
model = PneuSam(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)
model.train()
print(
    "Number of total parameters: ",
    sum(p.numel() for p in model.parameters()),
)  # 93735472
print(
    "Number of trainable parameters: ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)  # 93729252
print("args.dir_dataset: ", args.dir_dataset)
modelParams = list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters())
optimizer = torch.optim.AdamW(
        modelParams, lr=args.lr, weight_decay=args.weight_decay
    )
print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in modelParams if p.requires_grad),
    )
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
num_epochs = args.num_epochs
iter_num = 0
losses = []
val_losses = []
best_loss = 1e10
best_val_loss = 1e10
train_dataset = PneumothoraxDataset(args.dir_dataset)
val_dataset = PneumothoraxDataset(args.dir_dataset, image_set="val")
sampler = PneumoSampler(args.dir_dataset)
print("Number of training samples: ", len(train_dataset))
print("Number of validation samples: ", len(val_dataset))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sampler=sampler,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
)
start_epoch = 0
print("num_epochs: ", num_epochs)

for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0
    pbar = tqdm(train_dataloader, desc=f"Train on epoch {epoch}")
    for step, (image, gt2D, boxes, _) in enumerate(pbar):
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        pred = model(image, boxes_np)
        loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        iter_num += 1

    epoch_loss /= step
    pbar.set_postfix(loss=loss.item())
    losses.append(epoch_loss)
    checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
    torch.save(checkpoint, os.path.join(model_save_path, "sam_model_latest.pth"))
    if epoch % 10 == 0:
        pBarVal = tqdm(val_dataloader, desc=f"Val on epoch {epoch}")
        val_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(pBarVal):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            pred = model(image, boxes_np)
            loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
            val_loss += loss.item()
        val_loss /= step
        pBarVal.set_postfix(loss=loss.item())
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "sam_model_val_best.pth"))
        
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(model_save_path, "sam_model_train_best.pth"))

plt.plot(losses)
plt.title("Dice + Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
plt.close()

plt.plot(val_losses)
plt.title("Dice + Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(model_save_path, args.task_name + "val_loss.png"))
plt.close()