import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny

import cv2
from PIL import Image
import streamlit as st

MODEL_PATH = 'model/250218_base-model_ep-30.h5'
# MODEL_PATH = 'model/250218_augm-model_ep-50.h5'

@st.cache_resource
def load_model():
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
    
    return model

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

def predict_image(image, model):
    input_tensor = image_to_tensor(image)
    prediction = model.predict(input_tensor)
    probability = prediction[0][1] # 확률값
    
    if probability>=0.5:
        return round(probability*100, 2), '갑상선 암'
    else:
        return round((1-probability)*100, 2), '정상'