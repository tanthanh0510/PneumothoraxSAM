from flask import Flask, request
from flask_cors import CORS

import numpy as np

import torch

from skimage import io, transform

from segment_anything import sam_model_registry
from segment_anything.model import PneuSam

from utils.mask_binarizers import TripletMaskBinarization

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "experiment/sam_model_val_best.pth", map_location=device)


# set up model
orgiSam = sam_model_registry["vit_b"](
    checkpoint="/home/thanhtran/workdir/project/finetune-anything/sam_ckpt/sam_vit_b_01ec64.pth").to(device)
model = PneuSam(orgiSam.image_encoder, orgiSam.mask_decoder,
                orgiSam.prompt_encoder)
checkpoint = torch.load(
    'experiment/sam_model_val_best.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
print("Model loaded at epoch", checkpoint['epoch'])
triples = [list(checkpoint.get("best_threshold", (0.75, 1000, 0.4)))]
mask_binarizer = TripletMaskBinarization(triples)

model.eval()


def predict(path):
    img_np = io.imread(path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    img = transform.resize(img_3c, (1024, 1024), order=3,
                           preserve_range=True, anti_aliasing=True).astype(np.uint8)

    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    img = torch.tensor([img]).float()
    box_np = np.array([1, 0, W, H])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    box_1024 = torch.tensor([box_1024]).long()

    print(img.shape, box_1024.shape)
    pred = model(img, box_1024)
    pred = torch.sigmoid(pred)
    mask_generator = mask_binarizer.transform(pred)
    list_masks = []
    for curr_threshold, curr_mask in zip(mask_binarizer.thresholds, mask_generator):
        curr_mask = curr_mask.cpu().numpy()
        list_masks.append((curr_threshold, curr_mask))

    return list_masks


@app.route('/predict', methods=['POST'])
def get_prediction():
    # revice image from request and save to disk
    local_path = "test.png"
    img = request.files['image']
    img.save(local_path)
    masks = predict(local_path)
    for mask in masks:
        print(mask[0])
        m = mask[1]
        # get image from m
        m = m[0] * 255
        m = m.astype(np.uint8)

        print(m.shape)
        # save image to disk
        io.imsave("mask.png", m)

    # call predict function
    # return result
    return "OK"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
