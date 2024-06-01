import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from ultralytics import YOLO


def preproces(image_path):
    '''
    Читает изображение из пути, переводит в серое с 3 каналами
    '''
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image.shape
    return cv2.merge((gray_image, gray_image, gray_image))


def load_model(path):
    model = YOLO(path)
    return model

def predict_results(model, path):
    '''
    Предсказывает результат детекции модели для изобр. в пути
    '''
    image = preproces(path)
    return model(image)

def get_scale_img(model, image_path):
    '''
    Из результатов детекции получает координаты и вырезает часть со шкалой
    из изображения
    '''
    detection_results = predict_results(model, image_path)
    result = detection_results[0]
    box_coord = result.boxes.xyxy.cpu().numpy()#.squeeze()
    box_coord = box_coord.round().astype(int)
    (x_min, y_min, x_max, y_max) = box_coord[0]
    # Вырезаем часть со шкалой
    scale_bar = result.orig_img[y_min - 5:y_max - 5, x_min - 5:x_max +  5, :]
    scale_bar = cv2.cvtColor(scale_bar, cv2.COLOR_BGR2GRAY)
    return scale_bar