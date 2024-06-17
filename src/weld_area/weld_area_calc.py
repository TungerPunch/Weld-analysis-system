import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
sys.path.append('autoscale')
#from find_bar_functions import load_model, get_scale_img
from autoscale.autoscale import autoscale
from weld_area.weld_area_segment import get_segment_masks

def get_weld_area(image_path):
    '''
    Считает площадь сварного шва
    '''
    # Длина шкалы в м
    scale = autoscale(image_path)
    # Получение масок
    masks =  get_segment_masks(image_path)
    # Площадь пикселя
    pixel_area = scale**2
    # Расчет площади
    area = masks.sum() * pixel_area
    area = format(area, '.2e')
    return area, masks

def draw_mask(image, masks_generated) :
    masked_image = image.copy()

    for mask in masks_generated:
        mask = np.repeat(mask[:, :, None], 3, axis=2)
        print(mask.shape)
        masked_image = np.where(mask.astype(int),
                                np.array([0, 255, 0], dtype='uint8'),
                                masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def put_text_image(image, text):
    '''
    Отрисовывает текст с площадью на изображении
    '''
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 2
    fontColor              = (0,255, 0)
    thickness              = 2
    lineType               = 1
    cv2.putText(image, text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)

    return image


def draw_result(image_path):
    '''
    Отрисовывает маску и площадь
    '''
    area, masks = get_weld_area(image_path)
    image = cv2.imread(image_path)
    result_image = put_text_image(image, area + ' mm^2')
    result_image = draw_mask(image, masks)

    return result_image

