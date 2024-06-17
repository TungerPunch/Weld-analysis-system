from src.autoscale.autoscale import autoscale
from src.find_parameters.find_parameters import Parameters_finder
from ultralytics import YOLO

model = YOLO(r'models\find_weld_area.pt')


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

def inference(image):
    '''
    Функция для инференса
    '''
    mm_in_pxl = autoscale(image)
    PF = Parameters_finder(image=image,
                            model=model,
                            mm_in_pixel=mm_in_pxl)
    parameters = PF.find_all_parameters()
    parameters = [round(param, 3) for param in parameters]
    t, disp, width, depth, sagging, *undercuts, area = parameters
    
    keys = ['Width', 
            'Depth', 
            'Displacement', 
            'Q1',
            'Sagging',
            'Q2',
            'Undercut',
            'Q3',
            'Area']
    
    
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
    values = [
        width,
        depth,
        disp, 
        qual_level_disp,
        sagging,
        qual_level_sag,
        max(undercuts),
        qual_level_undercut,
        area
        ]
    result = dict(zip(keys, values))

    return result