import cv2
from PIL import Image
import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny

import torch
import torchvision.transforms as T
from ultralytics import YOLO

MODEL_PATH = 'model/250218_base-model_ep-30.h5'
YOLO_PATH = 'model/final/250319_yolov8_ep100/weights/best.pt'

@st.cache_resource
def load_model(diagnosis_type):
    if diagnosis_type==0:
        conv_model = ConvNeXtTiny(
            include_top=False,
            input_shape=(180, 180, 3),
            weights='imagenet'
        )
        model = models.Sequential([
            conv_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])

        model.load_weights(MODEL_PATH) # 가중치 로드
    else:
        model = YOLO(YOLO_PATH)

    return model

def image_to_tensor(image, img_size=(180, 180)):
    try:
        # 전처리 과정에서 그레이스케일 이미지 얻기
        prep_image = preprocess_image(image)
        prep_image = cv2.resize(prep_image, img_size)
        
        # 그레이스케일 -> 3채널 변환 (모델이 3채널 입력 기대)
        prep_image_rgb = np.stack([prep_image, prep_image, prep_image], axis=-1)
        prep_image_rgb = np.expand_dims(prep_image_rgb, axis=0)  # 배치 차원 추가
        
        return tf.convert_to_tensor(prep_image_rgb, dtype=tf.float32)
    except Exception as e:
        print(f'Error in image_to_tensor: {e}')
        print(f'Image shape: {image.shape if hasattr(image, "shape") else "Unknown"}')
        raise

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image) # PIL 이미지를 NumPy 배열로 변환

    # 채널 확인 및 처리
    if len(image.shape)==2:  # 이미 그레이스케일
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif len(image.shape)==3:
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        # 채널 순서 확인 (RGB/BGR 처리)
        if image.shape[2]==3:  # 컬러 이미지
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = blurred_image[:, :, 0]  # 다중 채널이지만 3채널이 아닐 경우 첫 채널 사용
    else:
        raise ValueError('Unexpected image format')
    
    if 'gray_image' not in locals():
        gray_image = blurred_image
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    
    return enhanced_image  # 흑백 이미지 (H, W)

def preprocess_image_for_yolo(image, size=(512, 512)):
    if isinstance(image, Image.Image): # PIL 이미지를 NumPy 배열로 변환
        image = np.array(image)

    # 그레이스케일이면 RGB로 변환
    if len(image.shape)==2 or image.shape[2]==1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # CLAHE 적용
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 크기 조정
    resized = cv2.resize(enhanced, size)

    return Image.fromarray(resized)  # YOLO는 PIL도 받음

def predict_image(image, model, diagnosis_type):
    if diagnosis_type==0:
        input_tensor = image_to_tensor(image)
        prediction = model.predict(input_tensor)
        probability = prediction[0][1] # 확률값
        
        if probability>=0.5:
            return round(probability*100, 2), '갑상선 암'
        else:
            return round((1-probability)*100, 2), '정상'
    else:
        preprocessed_image = preprocess_image_for_yolo(image)
        results = model(preprocessed_image)
        detections = results[0].boxes

        if detections is None or len(detections)==0:
            return 0.0, '정상', preprocessed_image

        # 클래스 기반 판단
        boxes = detections.xyxy.cpu().numpy()
        scores = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy().astype(int)

        tumor_scores = [s for s, c in zip(scores, classes) if c == 1]  # 뇌종양 클래스 ID=1

        # 시각화 이미지
        visualized = results[0].plot()
        visualized_rgb = cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB)
        visualized_pil = Image.fromarray(visualized_rgb)

        if len(tumor_scores)==0:
            max_neg_score = max(scores) if len(scores)>0 else 0.0
            return round(max_neg_score*100, 2), '정상', visualized_pil
        
        max_tumor_score = max(tumor_scores)
        return round(max_tumor_score*100, 2), '뇌종양', visualized_pil