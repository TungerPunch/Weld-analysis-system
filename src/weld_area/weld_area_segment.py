import cv2
import numpy as np
import torch
from ultralytics import YOLO

model = YOLO(r'weld_area\best.pt')

def preproces_image(image_path):
    '''
    Читает изображение из пути, переводит в серое с 3 каналами
    '''
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.merge((gray_image, gray_image, gray_image))

def get_indexes_of_central_boxes(boxes):
    '''
    Вычисляем центральные позиции для каждой детектированного объекта, сохраняет их индексы,
    что в центре
    '''
    indexes_central = []
    # x, y -  центры масок
    for i, (x, y, w, h) in enumerate(boxes.xywhn.numpy()):
        center_position = np.array([x, y])
        if ((center_position > 0.35) & (center_position < 0.65)).all():
            indexes_central.append(i)
    return indexes_central

def resize_masks(masks, orig_shape):
    resized_masks = []
    height, width = orig_shape[:2]
    for mask in masks:
        resized_mask = cv2.resize(mask, (width, height),
                                  interpolation = cv2.INTER_AREA)
        resized_masks.append(resized_mask)
    resized_masks = np.array(resized_masks)

    return resized_masks

def keep_largest_contour(mask):
    # Находим контуры в маске
    mask = np.array(mask, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Если контуры не были найдены, возвращаем пустую маску
        return np.zeros_like(mask)

    # Находим самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)

    # Создаем новую маску и рисуем на ней только самый большой контур
    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, [largest_contour], -1, 1, -1)  # -1 означает, что мы рисуем внутри контура

    return new_mask

def postproces_masks(results):
    '''Постпроцессинг для масок:
    Принимает на вход выход Yolo модели
    1) Для каждой маски оставляем самый большой контур
    2) Вычисляем центральные позиции для каждой маски, сохраняет те,
    что в центре
    3) Resize к оригинальному размеру
    На выходе набор центральных масок оригинального размера np.array
    '''
    result = results[0].cpu()
    orig_shape = result.orig_img.shape
    boxes = result.boxes
    masks = result.masks.data.cpu().numpy()
    # Оставляю самый большой контур
    masks = np.array(list(map(keep_largest_contour, masks)))
    # Оставляю только центральные маски
    indexes = get_indexes_of_central_boxes(boxes)
    masks = [masks[i] for i in indexes]
    # Возвращаю маски к разумеру изображения
    result_masks = resize_masks(masks, orig_shape)

    return result_masks

def get_segment_masks(image_path):
    '''
    Получает на вход модель и путь к изображению
    На выходе маски np.array
    '''
    image = preproces_image(image_path)
    results = model(image)
    result_masks = postproces_masks(results)

    return result_masks

def get_result_boxes(image_path):
    '''
    Принимает путь изображения
    Возвращает координаты итоговых boxes в формате xyxy
    '''
    image = preproces_image(image_path)
    results = model(image)
    result = results[0].cpu()
    boxes = result.boxes
    indexes = get_indexes_of_central_boxes(boxes)
    boxes_coord = boxes.xyxy.numpy()
    result_coord = [boxes_coord[i] for i in indexes]
    
    return  result_coord