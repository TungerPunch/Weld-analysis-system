import glob
import pandas as pd
from inference import inference


def main():
    '''
    Функция для запуска программы
    '''

    pathes = glob.glob('Data\\actual\\*.jpg')
    
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
    problem_pathes = []
    for path in pathes[:]:
        try:
            row = inference(path)
            print('Done', path)
            df.loc[len(df)] = row.loc[0]
        except Exception as e:
            print(path, e, sep='\n')
            problem_pathes.append(path)
    
    print(problem_pathes)
    df.to_csv('result/auto_parameters2.csv')

if __name__ == '__main__':
    main()