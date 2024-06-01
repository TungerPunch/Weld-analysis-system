from autoscale.autoscale import autoscale
from find_parameters.find_parameters import Parameters_finder
from ultralytics import YOLO
import pandas as pd

model = YOLO(r'weld_area\best.pt')


def find_quality_level(t, h, b_coef, c_coef, d_coef):
    '''
    Исходя из параметров нахождение уровня качества шва
    '''
    if h <= b_coef * t:
        quality_level = 'B'
    elif h <= c_coef * t:
        quality_level = 'C'
    elif h <= d_coef * t:
        quality_level = 'D'
    else:
        quality_level = 'Дефект'
    
    return quality_level

def inference(path):
    '''
    Функция для инференса
    '''
    mm_in_pxl = autoscale(path)
    PF = Parameters_finder(image_path=path,
                            model=model,
                            mm_in_pixel=mm_in_pxl)
    parameters = PF.find_all_parameters()
    parameters = [round(param, 3) for param in parameters]
    t, disp, width, depth, sagging, *undercuts, area = parameters
    
    columns = ['Путь',
               'Ширина', 
               'Глубина', 
               'Линейное смещение', 
               'Уровень качества ',
               'Незаполненная разделка кромок',
               'Уровень качества 2',
               'Подрез',
               'Уровень качества 3']
    df = pd.DataFrame(columns=columns)
    
    qual_level_disp = find_quality_level(t, 
                                      disp, 
                                      0.1, 
                                      0.15,
                                      0.25)
    qual_level_sag = find_quality_level(t,
                                        sagging,
                                        0.1,
                                        0.2,
                                        0.3)
    qual_level_undercut = find_quality_level(t,
                                             max(undercuts),
                                             0.05,
                                             0.1,
                                             0.15)
    df.loc[0] = [path,
                 width,
                 depth,
                 disp, 
                 qual_level_disp,
                 sagging,
                 qual_level_sag,
                 max(undercuts),
                 qual_level_undercut
                ]

    return df

#print(inference(r'data\actual\13c.jpg'))