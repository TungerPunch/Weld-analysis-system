import glob
import pandas as pd
from src.inference import inference
import numpy as np
import flask
from PIL import Image
import io
from flask import Flask
app = Flask(__name__)
import json


# Метод для использования вручную

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            return flask.redirect(flask.request.url)
        file = flask.request.files.get('file')
        if not file:
            return 'There is no file!'
        # Читаем файл
        image_bytes = file.read()
        # Трасформируем к массиву
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        # Находим результат
        result = inference(image)
        return flask.jsonify(result)
    else:
        return flask.render_template('index.html')

# Метод для подключения по API

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        image = np.array(Image.open(io.BytesIO(image)))

        result = inference(image)

        data["response"] = result
        data["success"] = True
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)