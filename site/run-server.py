# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory, redirect

import os
import numpy as np
# from matplotlib import pyplot as plt
import skimage
import skimage.io
import skimage.filters
import skimage.transform
import skimage.feature
import numpy.linalg as sla
# from werkzeug.utils import secure_filename
# %matplotlib inline
from keras.layers import Input
from keras.applications import ResNet50  # resnet50

app = Flask(__name__)
app.config['facesFolder'] = 'static/img/faces/'
app.config['pokesFolder'] = 'static/img/pokes/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resizePhoto(img):
    shape = list(img.shape)
    if shape[0] < shape[1]:
        size = shape[0]
        x = (shape[1] - size) // 2
        img = img[0:size, x:x + size]
    else:
        size = shape[1]
        img = img[0:size, 0:size]
    img = skimage.transform.resize(img, (224, 224, 3))
    return img


def maybeMakeSmaller(img):
    shape = list(img.shape)
    if shape[0] > 370:
        coef = 370 / shape[0]
        img = skimage.transform.rescale(img, coef)
    if shape[1] > 550:
        coef = 550 / shape[1]
        img = skimage.transform.rescale(img, coef)
    if shape != img.shape:  # if image was rescaled
        skimage.io.imsave(app.config['facesFolder'] + str(id) + '.jpg', img)


id = 0


def getId():
    global id
    id += 1
    return id


def initId():
    global id
    id = len(os.listdir("static/img/faces"))
    # path, dirs, files = os.walk("static/img/faces").next()
    # id = len(files)


@app.route('/pokemon')
def main():
    print("it is simple main page")
    return render_template("main.html")


@app.route('/getpokemon', methods=['GET'])
def redirectToMain():
    if "id" not in request.args:
        return redirect("http://olimp-union/pokemon", code=302)
    id = request.args.get('id')
    ctx_params = {'id': id}
    return render_template(
        "getpokemon.html",
        **ctx_params)


@app.route('/getpokemon', methods=['POST'])
def getPokemon():
    if 'imagefile' not in request.files:
            return "no file found"
    file = request.files['imagefile']
    if file.filename == '':
        return 'No selected file'
    if not allowed_file(file.filename):
        return "Wrong file format. Please, try another file http://olimp-union.com/pokemon"
    id = getId()
    file.save(os.path.join(app.config['facesFolder'], str(id) + ".jpg"))

    faces = list()
    img = skimage.io.imread(app.config['facesFolder'] + str(id) + '.jpg')  # load image from file
    maybeMakeSmaller(img)
    img = resizePhoto(img)
    img = skimage.color.rgb2gray(img)  # make it gray
    img = skimage.color.gray2rgb(img)
    faces.append(img)
    data = np.array(faces).astype(np.float64)
    data = preprocess_input(data)  # classify an image
    faces_len = 1
    prediction_faces = resnet.predict(data).reshape(faces_len, 2048)

    print("Count distances...")
    distances = np.zeros((faces_len, poke_len))
    for face_idx, face in enumerate(prediction_faces):  # this is 1 face
        for poke_idx, poke in enumerate(prediction_poke):
            distances[face_idx, poke_idx] = np.dot(face, poke) / sla.norm(poke) / sla.norm(face)

    image_pairs = []
    idx = 0
    priority = np.argsort(distances[idx])[::-1]  # sort distances
    for i, p in enumerate(priority):
        poke_idx = p
        break
    image_pairs.append((idx, poke_idx))

    poke_idx = image_pairs[0][1]
    skimage.io.imsave(app.config['pokesFolder'] + str(id) + ".png", origin_pokemons[poke_idx])
    return redirect("http://olimp-union.com/getpokemon?id=" + str(id), code=302)


if __name__ == '__main__':
    input_tensor = Input(shape=(224, 224, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    from keras.applications.imagenet_utils import preprocess_input
    initId()

    print("Сatching pokemons...")
    totalAmount = len(os.listdir("pokefaces"))
    pokemons = list()
    origin_pokemons = list()
    dir_ = 'pokefaces'
    i = 0
    for path in os.listdir(dir_):
        i = i + 1
        path = os.path.join(dir_, path)
        if '.png' not in path:
            continue
        pokeimg = skimage.io.imread(path)  # load image from file
        origin_pokemons.append(skimage.io.imread(path))  # remember origin imgs
        pokeimg = skimage.transform.resize(pokeimg, (224, 224, 3))
        pokeimg = skimage.color.rgb2gray(pokeimg)
        pokeimg = skimage.color.gray2rgb(pokeimg)
        pokemons.append(pokeimg)
        if i % 10 == 0:
            print(str(i) + " of " + str(totalAmount) + " pokemons processed")
    print("All pokemons processed")
    data = np.array(pokemons).astype(np.float64)
    data = preprocess_input(data)
    poke_len = len(data)
    print("Resnet is making predictions. It may take few minutes...")
    prediction_poke = resnet.predict(data).reshape(poke_len, 2048)

    # np.random.shuffle(image_pairs)
    app.run()
