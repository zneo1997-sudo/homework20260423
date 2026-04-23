import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F


# =========================
# FCN
# =========================
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleFCN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=16, stride=8, padding=4, bias=False
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.upscore(x)
        return x


def decode_mask(mask):
    color_map = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
    }
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[mask == cls] = color
    return color_mask


def load_fcn_model(model_path, device):
    model = SimpleFCN(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_fcn_prediction(model, image, device):
    image_resized = image.resize((128, 128))
    tensor = F.to_tensor(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    color_mask = decode_mask(pred)
    return Image.fromarray(color_mask)


# =========================
# Faster R-CNN
# =========================
def load_faster_rcnn_model(model_path, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, 3
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_faster_rcnn_prediction(model, image, device, score_thresh=0.5):
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    image_np = np.array(image).copy()
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)

    label_names = {1: "rectangle", 2: "circle"}

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{label_names.get(label, label)}:{score:.2f}", fill="white")

    return pil_img


# =========================
# Mask R-CNN
# =========================
def load_mask_rcnn_model(model_path, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box, 3
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, 3
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def apply_mask_overlay(image_np, mask, color):
    out = image_np.copy().astype(np.float32)
    alpha = 0.45
    for c in range(3):
        out[:, :, c] = np.where(
            mask,
            out[:, :, c] * (1 - alpha) + alpha * color[c],
            out[:, :, c]
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def run_mask_rcnn_prediction(model, image, device, score_thresh=0.5):
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    image_np = np.array(image).copy()

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output["masks"].cpu().numpy()

    label_names = {1: "rectangle", 2: "circle"}
    colors = {
        1: np.array([255, 0, 0], dtype=np.uint8),
        2: np.array([0, 255, 0], dtype=np.uint8)
    }

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < score_thresh:
            continue
        mask_bin = mask[0] > 0.5
        image_np = apply_mask_overlay(image_np, mask_bin, colors.get(label, np.array([255, 255, 0], dtype=np.uint8)))

    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{label_names.get(label, label)}:{score:.2f}", fill="white")

    return pil_img