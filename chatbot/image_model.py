import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

MODEL_PATH = 'chatbot/250218_base-model_ep-30.h5'

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    
    return model

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image) # PIL 이미지를 NumPy 배열로 변환

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0) # 노이즈 제거
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변환
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE로 대비 향상
    enhanced_image = clahe.apply(gray_image)

    return enhanced_image # 흑백 이미지 (H, W)

def image_to_tensor(image, img_size=(180, 180)):
    prep_image = preprocess_image(image) # 전처리
    prep_image = cv.resize(prep_image, img_size) # 크기 조정
    prep_image = np.expand_dims(prep_image, axis=-1) # 차원 추가 -> (H, W, 1)
    prep_image = np.expand_dims(prep_image, axis=0) # 차원 추가 -> (1, H, W, 1)

    return tf.convert_to_tensor(prep_image, dtype=tf.float32)

def predict_image(image, model):
    input_tensor = image_to_tensor(image)
    prediction = model.predict(input_tensor)
    probability = prediction[0][0] # 확률값
    
    if probability>=0.5:
        return round(probability*100, 4), '갑상선 암' # 갑상선 암 예측
    else:
        return round((1-probability)*100, 4), '정상' # 정상 예측