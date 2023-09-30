import os
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.semantic_seg import PneumothoraxDataset

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

tr_dataset = PneumothoraxDataset("input/dataset1024",usePromt=0,isDemo=True)
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
rootImade = 'input/dataset1024/train'
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):

    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    # save the image image[idx].cpu().permute(1, 2, 0).numpy()
    pathImage1 = rootImade + '/' + names_temp[idx]
    os.system('cp ' + pathImage1 + ' ./result/image_test1.png')
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    pathImage2 = rootImade + '/' + names_temp[idx]
    os.system('cp ' + pathImage2 + ' ./result/image_test2.png')
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./result/data_test.png", bbox_inches="tight", dpi=300)
    plt.close()
    break
