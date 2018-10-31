import numpy as np
import tensorflow as tf

from scipy.misc import imread, imresize, imsave

import config

def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """

    im_arr = imread(image, mode='L')
    r, c = np.shape(im_arr)
    if c > input_width:
        c = input_width
        ratio = float(input_width / c)
        final_arr = imresize(im_arr, (int(32 * ratio), input_width))
    else:
        final_arr = np.zeros((32, input_width))
        ratio = float(32 / r)
        im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
        final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr, c

def label_to_array(label):
    try:
        label_array = np.zeros((25, config.NUM_CLASSES))
        for i in range(len(label)):
            try:
                label_array[i, config.CHAR_VECTOR.index(label[i])] = 1
            except Exception as ex:
                label_array[i, 0] = 1
        return label_array
    except Exception as ex:
        print(label)
        raise ex

def label_to_array_2(label):
    try:
        label_array = np.zeros((25))
        for i in range(len(label)):
            try:
                label_array[i] = config.CHAR_VECTOR.index(label[i])
            except Exception as ex:
                label_array[i] = 0
        return label_array
    except Exception as ex:
        print(label)
        raise ex

def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """

    try:
        return ''.join([config.CHAR_VECTOR[np.argmax(arr)] for arr in ground_truth if np.argmax(arr) < len(config.CHAR_VECTOR)])
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
