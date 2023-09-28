import argparse
import random
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset, get_sampler
from losses import get_losses
from extend_sam import get_model, get_optimizer, get_scheduler, get_runner
from utils.mask_binarizers import get_mask_binarizes
import numpy as np

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)

parser.add_argument(
    '--ckpt_path', default=None, type=str)
parser.add_argument('--dataset_dir', default=None, type=str)

parser.add_argument('--tensorboard_folder',
                    default=None, type=str)
parser.add_argument('--log_folder', default=None, type=str)

parser.add_argument('--model_folder', default=None, type=str)
parser.add_argument('--batch_size', default=None, type=str)
parser.add_argument('--cfg', default=None, type=str)


# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(
#         plt.Rectangle((x0, y0), w, h, edgecolor="blue",
#                       facecolor=(0, 0, 0, 0), lw=2)
#     )


if __name__ == '__main__':
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load(
            "./config/{task_name}.yaml".format(task_name=args.task_name))

    train_cfg = config.train
    sampler_cfg = train_cfg.get('sampler', None)
    val_cfg = config.val

    if args.dataset_dir:
        train_cfg.dataset.params.dataset_dir = args.dataset_dir
        val_cfg.dataset.params.dataset_dir = args.dataset_dir
        if sampler_cfg:
            sampler_cfg.params.dataset_dir = args.dataset_dir

    if args.ckpt_path:
        train_cfg.model.params.ckpt_path = args.ckpt_path

    if args.tensorboard_folder:
        train_cfg.tensorboard_folder = args.tensorboard_folder

    if args.log_folder:
        train_cfg.log_folder = args.log_folder

    if args.model_folder:
        train_cfg.model_folder = args.model_folder

    if args.batch_size:
        train_cfg.bs = int(args.batch_size)
        val_cfg.bs = int(args.batch_size)

    test_cfg = config.test

    train_dataset = get_dataset(train_cfg.dataset)
    # tr_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    #     print(image.shape, gt.shape, bboxes.shape)
    #     # show the example
    #     _, axs = plt.subplots(1, 2, figsize=(25, 25))
    #     idx = random.randint(0, 7)
    #     axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    #     show_mask(gt[idx].cpu().numpy(), axs[0])
    #     show_box(bboxes[idx].numpy(), axs[0])
    #     axs[0].axis("off")
    #     # set title
    #     axs[0].set_title(names_temp[idx])
    #     idx = random.randint(0, 7)
    #     axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    #     show_mask(gt[idx].cpu().numpy(), axs[1])
    #     show_box(bboxes[idx].numpy(), axs[1])
    #     axs[1].axis("off")
    #     # set title
    #     axs[1].set_title(names_temp[idx])
    #     # plt.show()
    #     plt.subplots_adjust(wspace=0.01, hspace=0)
    #     plt.savefig("./test.png", bbox_inches="tight", dpi=300)
    #     plt.close()
    #     break
    if sampler_cfg:
        sampler = get_sampler(sampler_cfg)

    # train_sampler = PneumoSampler(folds_distr_path, fold_id, non_empty_mask_prob)

    if sampler_cfg:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=train_cfg.num_workers,
                                  drop_last=train_cfg.drop_last, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=train_cfg.num_workers,
                                  drop_last=train_cfg.drop_last)
    val_dataset = get_dataset(val_cfg.dataset)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=val_cfg.num_workers,
                            drop_last=val_cfg.drop_last)
    losses = get_losses(losses=train_cfg.losses)
    # according the model name to get the adapted model
    model = get_model(model_name=train_cfg.model.sam_name,
                      **train_cfg.model.params)
    # opt_params = get_opt_pamams(model, lr_list=train_cfg.opt_params.lr_list, group_keys=train_cfg.opt_params.group_keys,
    #                             wd_list=train_cfg.opt_params.wd_list)
    # print('opt_params: ', opt_params)
    # print total parameters and trainable parameters in model
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    img_mask_encdec_params = list(model.mask_decoder.parameters())+list(
        model.image_encoder.parameters())
    optimizer = get_optimizer(opt_name=train_cfg.opt_name, params=img_mask_encdec_params,
                              lr=train_cfg.opt_params.lr_default, weight_decay=train_cfg.opt_params.wd_default)
    scheduler = get_scheduler(
        optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name)
    mask_binarizer_cfg = config.mask_binarizer
    mask_binarizer_fn = get_mask_binarizes(mask_binarizer_cfg)
    runner = get_runner(train_cfg.runner_name)(
        model, optimizer, losses, train_loader, val_loader, scheduler, mask_binarizer_fn)
    # train_step
    runner.train(train_cfg)
    if test_cfg.need_test:
        runner.test(test_cfg)
