import numpy as np
import cv2

import matplotlib.pyplot as plt
import sys
sys.path.append('weld_area')



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
    '''
    Функция для ресайза масок сегментации
    '''
    resized_masks = []
    height, width = orig_shape[:2]
    for mask in masks:
        resized_mask = cv2.resize(mask, (width, height),
                                interpolation = cv2.INTER_AREA)
        resized_masks.append(resized_mask)
    resized_masks = np.array(resized_masks)

    return resized_masks

def keep_largest_contour(mask):
    '''
    Функция для нахождения наибольшего контура и удаления остальных
    '''
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
    cv2.drawContours(new_mask, [largest_contour], -1, 255, -1)  # -1 означает, что мы рисуем внутри контура

    return new_mask

class Parameters_finder:
    '''
    Класс для нахождения параметров лазерной сварки
    '''

    def __init__(self, image_path, model, mm_in_pixel):
        self.image = cv2.imread(image_path)
        # yolo segmentation model results
        self.results = model(self.image)
        # autoscale model results
        self.mm_in_pixel = 1
        self.path = image_path
        

    def get_segment_masks(self):
        '''Постпроцессинг для масок:
        Принимает на вход выход Yolo модели
        1) Для каждой маски оставляем самый большой контур
        2) Вычисляем центральные позиции для каждой маски, сохраняет те,
        что в центре
        3) Resize к оригинальному размеру
        На выходе набор центральных масок оригинального размера np.array
        '''
        result = self.results[0].cpu()
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
    
    def get_result_boxes(self):
        '''
        Принимает путь изображения
        Возвращает координаты итоговых boxes в формате xyxy
        '''
        result = self.results[0].cpu()
        boxes = result.boxes
        indexes = get_indexes_of_central_boxes(boxes)
        boxes_coord = boxes.xyxy.numpy()
        result_coord = [boxes_coord[i] for i in indexes]

        return  result_coord
    
    def get_weld_area_image(self):
        '''
        Вырезает из исходного изображения часть, содержащую проплавленную область
        '''
        boxes_coord = self.get_result_boxes()
        if len(boxes_coord) == 1:
            # Убираем пустую размерность, округляем
            box_coord = boxes_coord[0]
            box_coord = box_coord.round().astype(int)
            (x_min, y_min, x_max, y_max) = box_coord
            # Вырезаем интересующую нас часть
            weld_image = self.image[:,
                            x_min-100:x_max+100,
                            :]
            weld_image = cv2.cvtColor(weld_image, cv2.COLOR_BGR2GRAY)
            
            return weld_image, (x_min, y_min, x_max, y_max)
        else:
            return None
    
    
    def find_weld_area_width(self):
        '''
        Находит ширину шва
        '''
        masks = self.get_segment_masks()
        # Если шов цельный - определяем параметр
        if len(masks) != 1:
            return None
        else:
            mask = masks[0]

        _, x = np.where(mask == 255)

        width_in_pxl = x.max() - x.min()
        width = self.mm_in_pixel * width_in_pxl

        return width

    def get_plate_thickness(self):
        '''
        Возвращает толщину пластины на изображении и координаты верхнего края и нижнего
        '''
        weld_image, _ = self.get_weld_area_image()
        _, binary = cv2.threshold(weld_image, 
                                thresh=127, 
                                maxval=255, 
                                type=cv2.THRESH_OTSU)
        # Оставляем только самый большой контур
        binary = keep_largest_contour(binary)
        # Нахожу толщину в пикселях с левой стороны и с правой, считаю среднее
        left_stripe = binary[:, :10]
        y, x = np.where(left_stripe==255)
        y_max_left = y.max()
        y_min_left = y.min()
        left_plate_thickness = y_max_left - y_min_left
        right_stripe = binary[:, -10:]
        y, x = np.where(right_stripe==255)
        y_max_right = y.max()
        y_min_right = y.min()
        right_plate_thickness = y_max_right - y_min_right
        plate_thickness_in_pxl = (right_plate_thickness + left_plate_thickness) / 2
        plate_thickness = plate_thickness_in_pxl * self.mm_in_pixel

        return plate_thickness, (y_max_left, y_max_right, y_min_left, y_min_right)
    
    def find_displacement(self):
        '''
        Нахождение величины линейное смещения'''
        masks = self.get_segment_masks()
        # Если шов цельный - определяем параметр
        if len(masks) != 1:
            return None
        else:
            mask = masks[0]
        # Получение обрезанной маски
        weld_image, (x_min, y_min, x_max, y_max) = self.get_weld_area_image()
        y_mean = round((y_max + y_min) / 2)
        mask = mask[y_min:y_mean,
                    x_min:x_max]
        # Обрезаем еще раз
        y, x = np.where(mask==255)
        y_max, y_min, x_max, x_min = y.max(), y.min(), x.max(), x.min()
        mask = mask[y_min:y_max,
                    x_min:x_max]
        # Вырезаем края и находим координаты
        _, x_shape = mask.shape
        strip_val = round(x_shape / 4)
        left_strip = mask[:, :strip_val]
        y, x = np.where(left_strip==255)
        y_left = y.min()
        right_strip = mask[:, x_shape-strip_val:x_shape]
        y, x = np.where(right_strip==255)
        y_right = y.min()
        disp = np.abs(y_left - y_right) * self.mm_in_pixel

        return disp, (y_left, y_right)

    def find_weld_depth(self, disp):
        '''
        Находит глубину провара
        '''
        weld_area_and_coord =self.get_weld_area_image()
        # Если нет соединения(зон проплава больше 1) возвращаем None
        if weld_area_and_coord is None:
            return None
        else:
            weld_area_image, _ = weld_area_and_coord
        # Применяем медианное размытие, kernel_size - степень размытия
        kernel_size = 25
        weld_area_image = cv2.medianBlur(weld_area_image, kernel_size)

        # Бинаризируем
        _, binary = cv2.threshold(weld_area_image, 
                                thresh=127, 
                                maxval=255, 
                                type=cv2.THRESH_OTSU
                                )
        # Оставляем только самый большой контур
        weld_area_mask = keep_largest_contour(binary)

        y, x = np.where(weld_area_mask == 255)
        # Заполняем изображение сверху значением 255
        y_slice = int(y.mean())
        middle = weld_area_mask[:,
                                50:-50].copy()
        middle[:y_slice, :] = 255
        #plt.imshow(middle)
        #plt.show()
        plate_thickness, (y_max_left, 
                          y_max_right, 
                          y_min_left, 
                          y_min_right) = self.get_plate_thickness()
        y_plate_max = (y_max_right + y_max_left) / 2
        y_plate_min = (y_min_right + y_min_left) / 2
        y, _ = np.where(middle == 0)
        # Находим координату точки глубины провара
        weld_depth_coord = y.min()
        weld_depth_pxl = weld_depth_coord - y_plate_min
        weld_depth = weld_depth_pxl * self.mm_in_pixel + disp / 2
        return weld_depth
    
    def find_sagging(self):
        '''
        Нахождение величины прогиба (в метрах)
        '''
        masks = self.get_segment_masks()
        # Если шов цельный - определяем параметр
        if len(masks) != 1:
            return None
        else:
            mask = masks[0]
        # Получение обрезанной маски
        weld_image, (x_min, y_min, x_max, y_max) = self.get_weld_area_image()
        y_mean = round((y_min + (y_max - y_min) / 3))
        mask = mask[y_min:y_mean,
                    x_min:x_max]
        # Обрезаем еще раз
        y, x = np.where(mask==255)
        y_max, y_min, x_max, x_min = y.max(), y.min(), x.max(), x.min()
        mask = mask[y_min:y_max,
                    x_min:x_max]
        # Обрезаем центр
        _, x_shape = mask.shape
        x_mean = round(x_shape / 2)
        slide = round(x_shape / 20)
        middle = mask[:, x_mean - slide:x_mean + slide]
        # Находим сэггинг
        y, x = np.where(middle==255)
        sag_coord = y.min()
        disp, (y_left, y_right) = self.find_displacement()
        y_plate = (y_left + y_right) / 2

        if y_plate >= sag_coord:
            sagging = 0
        else:
            sagging_in_pxl = sag_coord - y_plate
            sagging = sagging_in_pxl * self.mm_in_pixel

        return sagging
    
    def find_undercuts(self):
        '''
        Нахождение величины подреза
        '''
        masks = self.get_segment_masks()
        # Если шов цельный - определяем параметр
        if len(masks) != 1:
            return None
        else:
            mask = masks[0]
        # Получение обрезанной маски
        weld_image, (x_min, y_min, x_max, y_max) = self.get_weld_area_image()
        y_mean = round((y_min + (y_max - y_min) / 3))
        mask = mask[y_min:y_mean,
                    x_min:x_max]
        # Обрезаем еще раз
        y, x = np.where(mask==255)
        y_max, y_min, x_max, x_min = y.max(), y.min(), x.max(), x.min()
        mask = mask[y_min:y_max,
                    x_min:x_max]
        # Убираем края
        _, x_shape = mask.shape
        slide = round(x_shape / 4)
        middle = mask[:, slide - 20:x_shape - slide + 20]
        # Разделяем
        _, x_shape = middle.shape
        slide = round(x_shape / 4)
        left_part = middle[:, :slide]
        right_part = middle[:, x_shape-slide:]
        # Находим координаты краев пластин
        disp, (y_left, y_right) = self.find_displacement()
        # Создаем список для результатов
        undercuts = []
        for img, y_plate in ((left_part, y_left),
                              (right_part, y_right)):
            # Верхний уровень через max потому что отсчет по вертикали сверху
            y, x = np.where(img == 0)
            undercut_coord = y.max()
            # Если координата подреза выше, чем координата края, то подрез 0
            if y_plate >= undercut_coord:
                undercut = 0
            else:
                undercut_in_pxl = undercut_coord - y_plate
                undercut = undercut_in_pxl * self.mm_in_pixel
            undercuts.append(undercut)
        return undercuts
    
    def fing_weld_area(self):
        '''
        Считает площадь сварного шва
        '''
        # Получение масок
        masks =  self.get_segment_masks()
        if len(masks) > 1:
            return None
        else:
            mask = masks[0]
        #plt.imsave(self.path.replace('actual', 'masks'), mask)
        # Площадь пикселя
        pixel_area = self.mm_in_pixel**2
        # Расчет площади
        area = mask.sum() * pixel_area
        #area = format(area, '.2e')
    
        return area
    
    def find_all_parameters(self):
        '''
        Функция для нахождения всех параметров сразу
        '''
        thickness, _ = self.get_plate_thickness()
        disp, _ = self.find_displacement()
        width = self.find_weld_area_width()
        depth = self.find_weld_depth(disp)
        sagging = self.find_sagging()
        undercuts = self.find_undercuts()
        area = self.fing_weld_area()

        return thickness, disp, width, depth, sagging, *undercuts, area