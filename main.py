import PIL
from PIL import Image
import numpy as np
import math
import zlib
import cv2
import gzip
import io
import skimage.util
import sys
import bz2
import os
import struct
import time
import threading
import itertools
sv = math.sqrt(8 / 64)
sv2 = math.sqrt(2 / 4)
haar_wavelet = [sv, sv, 1 / 2, 0, sv2, 0, 0, 0,
                sv, sv, 1 / 2, 0, -sv2, 0, 0, 0,
                sv, sv, -1 / 2, 0, 0, sv2, 0, 0,
                sv, sv, -1 / 2, 0, 0, -sv2, 0, 0,
                sv, -sv, 0, 1 / 2, 0, 0, sv2, 0,
                sv, -sv, 0, 1 / 2, 0, 0, -sv2, 0,
                sv, -sv, 0, -1 / 2, 0, 0, 0, sv2,
                sv, -sv, 0, -1 / 2, 0, 0, 0, -sv2]

sortedIndices = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33,
                 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
                 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
                 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

haar_wavelet = np.reshape(haar_wavelet, (8, 8))
h_t = np.transpose(haar_wavelet)
command = sys.argv[2]
input_file = sys.argv[1]
output_file = sys.argv[3]
threshold = sys.argv[4]


def checkIfDivisible(width, height):
    if (width % 8 == 0 and height % 8 == 0):
        return 1
    else:
        return 0


def haar_transform(array):
    # result = np.dot(np.dot(h_t,array), haar_wavelet)
    result = h_t @ array
    result = result @ haar_wavelet
    return result


def inverse_haar_transform(array):
    result = np.dot(np.dot(haar_wavelet, array), h_t)
    return result


def padImage(width, height, filepath, img):
    new_width = 0
    new_height = 0
    while (height % 8 != 0):
        height = height + 1
    while (width % 8 != 0):
        width = width + 1

    print("Padding image")
    result = Image.new(img.mode, (width, height), (0, 0, 0))
    result.paste(img, (0, 0))
    result.save(filepath)

    img_new = Image.open(filepath)
    newWidth, newHeight = img_new.size

    return img_new


def arrayToZigZag(array):
    sortedIndicesArray = np.zeros((64))
    unraveled_chunk = np.ravel(array)

    T = 0
    for i in range(64):
        if unraveled_chunk[sortedIndices[i]] < int(threshold):
            sortedIndicesArray[i] = 0
        else:
            sortedIndicesArray[i] = unraveled_chunk[sortedIndices[i]]

    return list(sortedIndicesArray)


def reverseZigZag(array):
    sortedIndicesArray = np.zeros((64))
    for i in range(64):
        sortedIndicesArray[sortedIndices[i]] = array[i]
    sortedIndicesArray = np.reshape(sortedIndicesArray, (8, 8))
    return sortedIndicesArray
done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rIn progress! ' + c)
        sys.stdout.flush()
        time.sleep(0.1)


if command == 'c':
    img = Image.open(input_file)
    width, height = img.size
    if checkIfDivisible(width, height) != 1:
        print("Gotta pad it, chief!")
        img = padImage(width, height, input_file, img)

    height, width = img.size

    R, G, B = Image.Image.split(img)
    R = np.array(R)
    G = np.array(G)
    B = np.array(B)

    print(R.shape)
    images = cv2.imread(input_file)
    height, width, c = images.shape
    t = threading.Thread(target=animate)
    t.start()
    print("-----------Transformation------------")
    red = []
    green = []
    blue = []
    stack = []
    t = threading.Thread(target=animate)
    t.start()
    file_size = []
    file_size.append(os.path.getsize(input_file))
    file_size = np.asanyarray(file_size)
    print(os.path.getsize(input_file))
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            red.append(arrayToZigZag(haar_transform(R[i:i + 8, j:j + 8])))
            green.append(arrayToZigZag(haar_transform(G[i:i + 8, j:j + 8])))
            blue.append(arrayToZigZag(haar_transform(B[i:i + 8, j:j + 8])))
    print("-----------Compressing------------")


    np.savez_compressed(output_file, a=red, b=green, c=blue, d=file_size, e = images.shape)
    done = True
    print("-----------Done--------------")

if command == 'd':

    loaded = np.load(input_file + '.npz',allow_pickle=True)
    print(loaded)
    red = loaded['a']
    green = loaded['b']
    blue = loaded['c']
    file_size = loaded['d']
    a,b, c= loaded['e']
    rr = np.zeros((a, b))
    gg = np.zeros((a, b))
    bb = np.zeros((a, b))

    np.set_printoptions(suppress=True)
    counter = 0
    for i in range(0, a, 8):
        for j in range(0, b, 8):
            rr[i:i + 8, j:j + 8] = inverse_haar_transform(reverseZigZag(red[counter]))
            gg[i:i + 8, j:j + 8] = inverse_haar_transform(reverseZigZag(green[counter ]))
            bb[i:i + 8, j:j + 8] = inverse_haar_transform(reverseZigZag(blue[counter]))

            counter = counter + 1

    merged = cv2.merge([bb, gg, rr])
    cv2.imwrite(output_file, merged)
    curr_file_size = os.path.getsize(output_file)
    print("Compression rate: ", file_size / curr_file_size)
 #ab = list(itertools.chain(r, g, b))
    #buf1 = struct.pack('%sf' % len(r), *r)
   # r = list(itertools.chain.from_iterable(red))
  #  g = list(itertools.chain.from_iterable(green))
 #   b = list(itertools.chain.from_iterable(blue))
#    print(len(r))

    #buf2 = struct.pack('%sf' % len(g), *g)
    #buf3 = struct.pack('%sf' % len(b), *b)
