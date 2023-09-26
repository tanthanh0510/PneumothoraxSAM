from typing import DefaultDict
from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler, binarizer_fn=None):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        self.binarizer_fn = binarizer_fn
        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)


class SemRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler, binarizer_fn=None):
        super().__init__(model, optimizer, losses,
                         train_loader, val_loader, scheduler, binarizer_fn)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_score = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(
            cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(
            cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training on device {device}".format(device=device))
        writer = None
        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(
                cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)
        # train
        print("Start training")
        # print total parameters and trainable parameters in model
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        for iteration in range(cfg.max_iter):
            images, labels = train_iterator.get()
            images, labels = images.to(device), labels.to(device).long()
            masks_pred, iou_pred = self.model(images)
            masks_pred = F.interpolate(
                masks_pred, self.original_size, mode="bilinear", align_corners=False)
            # if self.model.num_classes == 1:
            masks_pred = masks_pred.view(-1,
                                         self.original_size, self.original_size)

            total_loss = torch.zeros(1).to(device)
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(iteration=iteration, log_path=log_path, log_data=train_meter.get(clear=True),
                          status=self.exist_status[0],
                          writer=writer, timer=self.train_timer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:
                metric, score = self._eval(iteration, writer, cfg)
                print(metric)
                if best_valid_score == -1 or best_valid_score < score:
                    best_valid_score = score
                    save_model(self.model, model_path,
                               parallel=self.the_number_of_gpu > 1)
                    print_and_save_log("saved model in {model_path}".format(
                        model_path=model_path), path=log_path)
                log_data = {'score': score,
                            'best_valid_score': best_valid_score}
                write_log(iteration=iteration, log_path=log_path, log_data=log_data, status=self.exist_status[1],
                          writer=writer, timer=self.eval_timer)
        # final process
        save_model(self.model, model_path, is_final=True,
                   parallel=self.the_number_of_gpu > 1)
        if writer is not None:
            writer.close()

    def test(self):
        pass

    def _eval(self, iteration, writer, cfg):
        self.model.eval()
        self.eval_timer.start()
        metrics = DefaultDict(float)

        used_thresholds = self.binarizer_fn.thresholds
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_eval_file.txt".format(
            cfg=cfg)
        check_folder(log_path)
        # eval_metric = mIoUOnline(class_names=class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.to(device)
                labels = labels.to(device)
                masks_pred, iou_pred = self.model(images)
                masks_pred = F.interpolate(
                    masks_pred, self.original_size, mode="bilinear", align_corners=False)
            # if self.model.num_classes == 1:
                masks_pred = masks_pred.view(-1,
                                             self.original_size, self.original_size)
                predictions = torch.sigmoid(masks_pred)
                mask_generator = self.binarizer_fn.transform(predictions)
                for curr_threshold, curr_mask in zip(used_thresholds, mask_generator):
                    curr_metric = self.eval_fn(curr_mask, labels).item()
                    curr_threshold = tuple(curr_threshold)
                    metrics[curr_threshold] = (
                        metrics[curr_threshold] * index + curr_metric) / (index + 1)

            best_threshold = max(metrics, key=metrics.get)
            tmp = {
                "result": f'Score: {metrics[best_threshold]:.5} at threshold {best_threshold}'}
            write_log(iteration=iteration, log_path=log_path, log_data=tmp,
                      status=self.exist_status[0],
                      writer=writer, timer=self.train_timer)
        self.model.train()
        self.eval_timer.end()
        return metrics, metrics[best_threshold]

    def eval_fn(self, preds, trues, per_image=True, per_channel=False):
        preds = preds.float()
        return 1 - soft_dice_loss(preds, trues, per_image, per_channel)

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels
            # if loss_cfg[item[0]].label_one_hot:
            #     class_num = cfg.model.params.class_num
            #     real_labels = one_hot_embedding_3d(
            #         real_labels, class_num=class_num)
            tmp_loss = item[1](mask_pred, real_labels)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]].weight * tmp_loss

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels
            # if loss_cfg[item[0]].label_one_hot:
            #     class_num = cfg.model.params.class_num
            #     real_labels = one_hot_embedding_3d(
            #         real_labels, class_num=class_num)
            tmp_loss = item[1](mask_pred, real_labels)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]].weight * tmp_loss


def soft_dice_loss(outputs, targets, per_image=False, per_channel=False):
    batch_size, n_channels = outputs.size(0), outputs.size(1)

    eps = 1e-6
    n_parts = 1
    if per_image:
        n_parts = batch_size
    if per_channel:
        n_parts = batch_size * n_channels

    dice_target = targets.contiguous().view(n_parts, -1).float()
    dice_output = outputs.contiguous().view(n_parts, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss
    # for batch_index in range(images.size()[0]):
    #     pred_mask = get_numpy_from_tensor(predictions[batch_index])
    #     gt_mask = get_numpy_from_tensor(
    #         labels[batch_index].squeeze(0))
    #     h, w = pred_mask.shape
    #     gt_mask = cv2.resize(
    #         gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    #     eval_metric.add(pred_mask, gt_mask)

    # return eval_metric.get(clear=True)

    # def val_epoch(self, model, loader):
    #     tqdm_loader = tqdm(loader)
    #     curr_score_mean = 0
    #     used_thresholds = self.binarizer_fn.thresholds
    #     metrics = DefaultDict(float)

    #     model.eval()
    #     with torch.no_grad():
    #         for batch_idx, (images, labels) in enumerate(tqdm_loader):
    #             pred_probs = self.batch_val(model, images)
    #             labels = labels.to(self.device)
    #             mask_generator = self.binarizer_fn.transform(pred_probs)

    #             for curr_threshold, curr_mask in zip(used_thresholds, mask_generator):
    #                 curr_metric = self.eval_fn(curr_mask, labels).item()
    #                 curr_threshold = tuple(curr_threshold)
    #                 metrics[curr_threshold] = (
    #                     metrics[curr_threshold] * batch_idx + curr_metric) / (batch_idx + 1)

    #             best_threshold = max(metrics, key=metrics.get)
    #             tqdm_loader.set_description(
    #                 f'Score: {metrics[best_threshold]:.5} at threshold {best_threshold}')

    #     return metrics, metrics[best_threshold]

    # def batch_val(self, model, batch_image):
    #     batch_image = batch_image.to(self.device)
    #     preds = model(batch_image)

    #     return torch.sigmoid(preds)
