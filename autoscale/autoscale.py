from autoscale.find_bar_functions import load_model, get_scale_img
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract
import re
# Указываю путь к приложению tesseract
pytesseract.pytesseract.tesseract_cmd = r'autoscale\tesseract\tesseract.exe'

model = load_model('yolo_find_bar\yolo_best.pt')

def get_binary_scale(scale):
    '''
    Бинаризирует шкалу, 
    возвращается белые надписи на черном фоне
    '''
    _, binary = cv2.threshold(scale, 
                              thresh=122, 
                              maxval=255, 
                              type=cv2.THRESH_OTSU)
    if binary.mean() > 127:
        _, binary = cv2.threshold(binary, 
                                  thresh=127, 
                                  maxval=255, 
                                  type=cv2.THRESH_BINARY_INV)
    return binary


def find_pixels_in_bar(binary_scale):
    '''
    Возвращает длину шкалы и и список с контурами букв текста
    '''
    contours, _ = cv2.findContours(image = binary_scale, 
                                    mode = cv2.RETR_EXTERNAL,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    
    contours = list(map(np.squeeze, contours))
    # Находим максимальную длину контуров по х, самая широкая - шкала
    lengths = []
    for contour in contours:
        try:
            _, _, w, _ = cv2.boundingRect(contour)
        except:
            continue
        lengths.append(w)
    # Нахожу длину шкалы в пикселях
    pixels_in_scale = max(lengths)

    return pixels_in_scale



def image_to_text(image_with_text):
    """ 
    Считывает буквы с контуров и собирает в 1 строку 
    """
    config = ("-l eng --oem 1 --psm 11")
    text = pytesseract.image_to_string(image_with_text, 
                                       config=config)
    return  text

def autoscale(image_path):
    '''
    Принимает изображение с микроскопа
    Возвращает количество миллиметров в пикселе
    '''
    # Находим и бинаризируем шкалу
    scale = get_scale_img(model, image_path)
    bin_scale = get_binary_scale(scale)
    # Находим количество пикселей в шкале и контуры букв
    pixels_in_bar = find_pixels_in_bar(bin_scale)
    # Находим текст
    text = image_to_text(bin_scale)
    text = re.sub('[\W_]+', '', text)
    text = text.replace('I', '1').replace('i', '1')
    # Разделяем текст на число и слово
    number_in_text, string_in_text = re.findall(r'(\d+)(\D+)', text)[0]
    number_in_text = float(number_in_text)
    # Убираем все пробелы
    string_in_text = re.sub('\s+', '', string_in_text)
    # Определяем размерность
    if string_in_text in ['мм', 'mm']:
        coef  = 1
    elif string_in_text in ['um', 'pm', 'мкм']:
        coef = 1e-3
    elif string_in_text in ['nm', ]:
        coef = 1e-6
    else:
        pass
        raise ValueError('Некорректно распознан текст')
    mm_in_pixel = coef * number_in_text / pixels_in_bar

    return mm_in_pixel