# Weld-analysis-system
Система автоматического анализа качества и геометрии сварного шва при лазерной сварке. Включает в себя автоматическое определение масштаба изображения, определение и вычисление площади сварного шва и анализ геометрии и дефектности сварного шва. Для системы создано веб-приложение и обернуто в докер контейнер.

Используемые технологии:
python,
ultralytics,
roboflow,
opencv,
numpy,
pytesseract,
flask,
docker,


## Для запуска веб-приложения:

```powershell 

python main.py

```

## Схема работы системы:

![image](https://github.com/TungerPunch/Weld-analysis-system/assets/86575050/94121e3c-4a2f-4899-8358-6d22ddfe3f6d)


Изображение 2592x1944 подается в 2 модели - детекции и инстанс-сегментации

#### Детекция масштабной шкалы
Модель детекции yolov8s определяет расположение масштабной шкалы на изображении. Далее задетектированная область бинаризуется, на ней распознается самый большой контур - масштабная шкала, и область подается в модель распознавания текста - tesseract ocr. Через длину шкалы и распознанный текст определяется коэффициент масштаба

Для модели детекции были вручную размечены изображения, применена аугментация (horizontal, vertical flip, random crop, brightness) и получен датасет в 154 изображения с помощью фреймворка roboflow. Датасет разделен на тренировочную и тестовую выборку в соотношении 4:1. Модель yolov8s обучена с помощью фреймворка ultralytics.

#### Сегментация зоны сварного шва
Модель инстанс-сегментации yolov8s определяет область зоны сварного шва на изображении. Далее маски, центры которых не входят в центральную область x 972:1620, y 729:1215, отсекаются, и на оставшихся масках оставляется только самый большой контур. Если на выходе 2 маски - фиксируется дефект непровар, соединения нет. Если маска 1, по ней расчитываются площадь, глубина, ширина, линейное смещение, незаполненная разделка кромок и подрез.

Для модели детекции были вручную размечены изображения, применена аугментация (horizontal flip, random crop, brightness) и получен датасет в 154 изображения с помощью фреймворка roboflow. Датасет разделен на тренировочную и тестовую выборку в соотношении 4:1. Модель yolov8s обучена с помощью фреймворка ultralytics.

#### Дефекты:
![image](https://github.com/TungerPunch/Weld-analysis-system/assets/86575050/4578da87-03a2-4ee4-b1e1-6962fd6f8c6c)
Линейное смещение
![image](https://github.com/TungerPunch/Weld-analysis-system/assets/86575050/c186e706-57fb-4e85-b5f1-ddd18fb9a1d0)
Незаполенная разделка кромок
![image](https://github.com/TungerPunch/Weld-analysis-system/assets/86575050/d2e4d306-eec3-4dce-ada2-f82f5dcf54ff)
Подрез

### Веб-приложение

Для системы написано веб-приложение с возможностью ручной загрузки изображений и подключения по API
![image](https://github.com/TungerPunch/Weld-analysis-system/assets/86575050/d1947f0d-8e7a-42cf-8726-2acde18935a6)

