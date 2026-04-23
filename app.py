import io
import os
import random
import numpy as np
from PIL import Image, ImageDraw

import streamlit as st
import torch
from torchvision.transforms import functional as F

from utils import (
    load_fcn_model,
    load_faster_rcnn_model,
    load_mask_rcnn_model,
    run_fcn_prediction,
    run_faster_rcnn_prediction,
    run_mask_rcnn_prediction,
)

st.set_page_config(page_title="Pattern Recognition Demo", layout="wide")

st.title("模式识别与图像处理课程作业演示系统")
st.markdown("### FCN / Faster R-CNN / Mask R-CNN 交互式演示")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"当前设备: `{device}`")

task = st.sidebar.selectbox(
    "选择功能",
    ["FCN语义分割", "Faster R-CNN目标检测", "Mask R-CNN实例分割"]
)

score_thresh = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.5, 0.05)

img_size = st.sidebar.selectbox("生成测试图尺寸", [128, 160, 256], index=1)

use_demo = st.sidebar.button("随机生成测试图")

uploaded_file = st.file_uploader("上传一张图片", type=["png", "jpg", "jpeg"])

def generate_demo_image(size=160):
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    num_objects = random.randint(1, 3)
    for _ in range(num_objects):
        shape_type = random.choice(["rectangle", "circle"])
        x1 = random.randint(5, size - 60)
        y1 = random.randint(5, size - 60)
        w = random.randint(25, 55)
        h = random.randint(25, 55)
        x2 = min(size - 5, x1 + w)
        y2 = min(size - 5, y1 + h)

        if shape_type == "rectangle":
            color = (
                random.randint(150, 255),
                random.randint(30, 120),
                random.randint(30, 120)
            )
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            color = (
                random.randint(30, 120),
                random.randint(150, 255),
                random.randint(30, 120)
            )
            draw.ellipse([x1, y1, x2, y2], fill=color)
    return image

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif use_demo:
    image = generate_demo_image(img_size)

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("输入图像")
        st.image(image, use_container_width=True)

    if task == "FCN语义分割":
        model_path = "models/best_fcn_model.pth"
        if not os.path.exists(model_path):
            st.error(f"未找到模型文件: {model_path}")
        else:
            model = load_fcn_model(model_path, device)
            result_img = run_fcn_prediction(model, image, device)

            with col2:
                st.subheader("分割结果")
                st.image(result_img, use_container_width=True)

    elif task == "Faster R-CNN目标检测":
        model_path = "models/faster_rcnn_fast_demo.pth"
        if not os.path.exists(model_path):
            st.error(f"未找到模型文件: {model_path}")
        else:
            model = load_faster_rcnn_model(model_path, device)
            result_img = run_faster_rcnn_prediction(model, image, device, score_thresh)

            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_container_width=True)

    elif task == "Mask R-CNN实例分割":
        model_path = "models/mask_rcnn_demo.pth"
        if not os.path.exists(model_path):
            st.error(f"未找到模型文件: {model_path}")
        else:
            model = load_mask_rcnn_model(model_path, device)
            result_img = run_mask_rcnn_prediction(model, image, device, score_thresh)

            with col2:
                st.subheader("实例分割结果")
                st.image(result_img, use_container_width=True)

            if result_img is not None:
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button(
                    "下载结果图",
                    data=buf.getvalue(),
                    file_name="result.png",
                    mime="image/png"
                )
else:
    st.info("请上传图片，或点击左侧“随机生成测试图”开始演示。")