import os
import argparse
import matplotlib.pyplot as plt

from typing import DefaultDict
from tqdm import tqdm
from datetime import datetime

import pandas as pd

import monai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.model import PneuSam
from datasets.semantic_seg import PneumothoraxDataset, PneumoSampler

from losses import dice_metric

from utils.mask_binarizers import TripletMaskBinarization
# set seeds
torch.manual_seed(49)
torch.cuda.empty_cache()


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir_dataset",
        type=str,
        default="input/dataset1024",
        help="dir dataset",
    )
    parser.add_argument("-mode", type=str, default="train")
    parser.add_argument("-usePromt", type=int, default=0)
    parser.add_argument("-valEpoch", type=int, default=10)
    parser.add_argument("-task_name", type=str, default="SAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-resume", type=str,
                        default=None)
    parser.add_argument(
        "-checkpoint", type=str, default="sam_ckpt/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./experiment")
    parser.add_argument("-num_epochs", type=int, default=500)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument(
        "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
    )
    parser.add_argument(
        "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument("-use_amp", action="store_true",
                        default=False, help="use amp")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


def getModel(args):
    bestLossTrain, bestLossVal = 1e10, 1e10
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = PneuSam(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(args.device)
    model.train()
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    modelParams = list(model.image_encoder.parameters()) + \
        list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        modelParams, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in modelParams if p.requires_grad),
    )
    startEpoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        startEpoch = checkpoint["epoch"]+1
        bestLossTrain = checkpoint.get("lossTrain", 1e10)
        bestLossVal = checkpoint.get("lossVal", 1e10)
        print("Loaded checkpoint from: ", args.resume)
        print("Loaded epoch: ", startEpoch)
        print("Loaded bestLossTrain: ", bestLossTrain)
        print("Loaded bestLossVal: ", bestLossVal)

    return model, optimizer, startEpoch, bestLossTrain, bestLossVal


def getDataLoaders(args):
    train_dataset = PneumothoraxDataset(args.dir_dataset, args.usePromt)
    val_dataset = PneumothoraxDataset(
        args.dir_dataset, args.usePromt, image_set="val")
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
    return train_dataloader, val_dataloader


def train(model, train_dataloader, seg_loss, ce_loss, optimizer, epoch, model_save_path, device, bestLoss):
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
        pbar.set_postfix(loss=loss.item())

    epoch_loss /= step
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "lossTrain": epoch_loss,
        "lossVal": 1e10,
    }
    torch.save(checkpoint, os.path.join(
        model_save_path, "sam_model_latest.pth"))

    if epoch_loss < bestLoss:
        bestLoss = epoch_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "lossTrain": bestLoss,
            "lossVal": 1e10,
        }
        torch.save(checkpoint, os.path.join(
            model_save_path, "sam_model_train_best.pth"))

    print(f"epoch_loss at {epoch}: {epoch_loss}, bestLoss: {bestLoss}")
    return epoch_loss, bestLoss


def process_summary(summary_file, metrics, epoch):
    best_threshold = max(metrics, key=metrics.get)

    epoch_summary = pd.DataFrame.from_dict([metrics])
    epoch_summary['epoch'] = epoch
    epoch_summary['best_metric'] = metrics[best_threshold]
    epoch_summary = epoch_summary[[
        'epoch', 'best_metric'] + list(metrics.keys())]
    epoch_summary.columns = list(map(str, epoch_summary.columns))

    print(
        f'Epoch {epoch + 1}\tScore: {metrics[best_threshold]:.5} at params: {best_threshold}')

    if not os.path.exists(summary_file):
        epoch_summary.to_csv(summary_file, index=False)
    else:
        summary = pd.read_csv(summary_file)
        summary = pd.concat([summary, epoch_summary]).reset_index(drop=True)
        summary.to_csv(summary_file, index=False)


def val(model, val_dataloader, seg_loss, ce_loss, optimizer, epoch, model_save_path, device, bestValLoss, maskBinarizer, bestScore):
    model.eval()
    valLoss = 0
    metrics = DefaultDict(float)
    pbar = tqdm(val_dataloader, desc=f"Val on epoch {epoch}")
    for step, (image, gt2D, boxes, _) in enumerate(pbar):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        pred = model(image, boxes_np)
        loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
        valLoss += loss.item()
        pred = torch.sigmoid(pred)
        mask_generator = maskBinarizer.transform(pred)
        for curr_threshold, curr_mask in zip(maskBinarizer.thresholds, mask_generator):
            curr_metric = dice_metric(curr_mask, gt2D)
            curr_threshold = tuple(curr_threshold)
            metrics[curr_threshold] = (
                metrics[curr_threshold] * step + curr_metric) / (step + 1)
        pbar.set_postfix(loss=loss.item(), curr_threshold=curr_threshold,
                         curr_metric=curr_metric.item())

    best_threshold = max(metrics, key=metrics.get)
    # write metrics to file
    valLoss /= step
    if metrics[best_threshold] > bestScore:
        bestScore = metrics[best_threshold]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "bestScore": bestScore,
            "lossVal": bestValLoss,
            "best_threshold": best_threshold
        }
        torch.save(checkpoint, os.path.join(
            model_save_path, "sam_model_val_best_score.pth"))

    if valLoss < bestValLoss:
        bestValLoss = valLoss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "lossTrain": 1e10,
            "lossVal": bestValLoss,
        }
        torch.save(checkpoint, os.path.join(
            model_save_path, "sam_model_val_best.pth"))
    process_summary(os.path.join(model_save_path,
                    "summary.csv"), metrics, epoch)
    print(f"valLoss at epoch {epoch}: {valLoss}, bestValLoss: {bestValLoss}")
    print(f'Score: {metrics[best_threshold]:.5} at threshold {best_threshold}')
    model.train()

    return valLoss, bestValLoss, bestScore


def test(args, maskBinarizer):
    model, _, _, _, _ = getModel(args)
    device = torch.device(args.device)
    model.eval()
    test_dataset = PneumothoraxDataset(
        args.dir_dataset, args.usePromt, image_set="val")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    pbar = tqdm(test_dataloader, desc=f"Test")
    totalScore = 0
    metrics = DefaultDict(float)
    for step, (image, gt2D, boxes, _) in enumerate(pbar):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        pred = model(image, boxes_np)
        pred = torch.sigmoid(pred)
        mask_generator = maskBinarizer.transform(pred)
        for curr_threshold, curr_mask in zip(maskBinarizer.thresholds, mask_generator):
            curr_metric = dice_metric(curr_mask, gt2D)
            curr_threshold = tuple(curr_threshold)
            metrics[curr_threshold] = (
                metrics[curr_threshold] * step + curr_metric) / (step + 1)
            pbar.set_postfix(curr_threshold=curr_threshold,
                             curr_metric=curr_metric.item())

    best_threshold = max(metrics, key=metrics.get)
    print(f'Score: {metrics[best_threshold]:.5} at threshold {best_threshold}')

    return totalScore


def main(args, maskBinarizer):
    bestScore = 0
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(
        args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    os.makedirs(model_save_path, exist_ok=True)

    model, optimizer, startEpoch, best_loss, best_val_loss = getModel(args)

    seg_loss = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    train_dataloader, val_dataloader = getDataLoaders(args)

    num_epochs = args.num_epochs
    losses = []
    val_losses = []
    print("num_epochs: ", num_epochs)

    for epoch in range(startEpoch, num_epochs):
        epoch_loss, best_loss = train(model, train_dataloader, seg_loss,
                                      ce_loss, optimizer, epoch, model_save_path, device, best_loss)
        losses.append(epoch_loss)
        if epoch % args.valEpoch == 0:
            val_loss, best_val_loss, bestScore = val(model, val_dataloader, seg_loss,
                                                     ce_loss, optimizer, epoch, model_save_path, device, best_val_loss, maskBinarizer, bestScore)
            val_losses.append(val_loss)

    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_save_path,
                args.task_name + "train_loss.png"))
    plt.close()

    plt.plot(val_losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_save_path, args.task_name + "val_loss.png"))
    plt.close()


if __name__ == "__main__":
    args = getArgs()
    triplets = [
        [0.75, 1000, 0.3],
        [0.75, 1000, 0.4],
        [0.75, 2000, 0.3],
        [0.75, 2000, 0.4],
        [0.6, 2000, 0.3],
        [0.6, 2000, 0.4],
        [0.6, 3000, 0.3],
        [0.6, 3000, 0.4],
    ]
    maskBinarizer = TripletMaskBinarization(triplets=triplets)
    if args.mode == "train":
        main(args, maskBinarizer)
    elif args.mode == "test":
        test(args, maskBinarizer)
    else:
        print("Invalid mode")
