import os
import skimage
import skimage.io
import skimage.transform

for path in os.listdir("unprocessedPokes"):
    pathWithUnprocessed = "unprocessedPokes/" + path
    pathWithProcessed = "pokefaces/" + path
    # print("path:" + pathWithProcessed)
    img = skimage.io.imread(pathWithUnprocessed)  # load image from file
    shape = list(img.shape)
    if shape[0] < shape[1]:
        size = shape[0]
        x = (shape[1] - size) // 2
        img = img[0:size, x:x + size]
    else:
        size = shape[1]
        img = img[0:size, 0:size]
    img = skimage.transform.resize(img, (224, 224))
    print(path)
    skimage.io.imsave(pathWithProcessed, img)

# return img
