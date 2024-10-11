#!/usr/bin/env python3

# 2020-2023 (c) Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
# SLS Team, TIMA Lab, Grenoble INP/UGA
# 
# This program is free software; you can redistribute it and/or modify it
# under the terms and conditions of the GNU General Public License,
# version 2 or later, as published by the Free Software Foundation.
# 
# This program is distributed in the hope it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
# 
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

# Dumping C arrays for the parameters learned in TensorFlow

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
from typing import List, Callable, Optional
import time
import os
import pathlib
import sys
import re

if len(sys.argv) != 2:
    print(f'Usage {sys.argv[0]} (float|int8_t)')
    sys.exit(1)

if  sys.argv[1] == "int8_t" or sys.argv[1] == "float":
    rep = sys.argv[1]
else:
    print(f'Usage {sys.argv[0]} (float|int8_t)')
    sys.exit(1)

# Find a fixed point multiplication
# Monkey-style translation from tensorflow gemmlowp pipeline
def mult_to_fix_shift(multiplicand):
    assert multiplicand > 0.0, "Multiplicand must be positive"
    assert multiplicand < 1.0, "Multiplicand must be strictly less than 1"

    rs = 0
    while multiplicand < 0.5:
        multiplicand *= 2.0
        rs += 1

    m = int(round(multiplicand * 2 ** 31))

    if m == 2 ** 31:
        return m / 2, rs - 1
    else:
        return m, rs

    return m, rs

# Stuff learned using tf on lenet.py
model_filename = f'lenet5_models/model_{rep}.tflite'
interpreter = tf.lite.Interpreter(model_path = model_filename)
interpreter.allocate_tensors()

# Ensure dump of the whole content of a numpy array
np.set_printoptions(threshold = np.inf)

# Dumps the whole stuff in sequential order
# For debug and understanding only, thus not executed in production mode
if False:
    # Gives the tensors in what seems to be the correct order, alleluhia!
    ops = interpreter._get_ops_details()
    for op in ops:
        print(op)
        index = 0
        for layer_idx in op['inputs']:
            layer = interpreter._get_tensor_details(layer_idx)
            scales_in = layer['quantization_parameters']['scales']
            print(f'Input[{index}] ', layer['name'], layer['quantization_parameters']['scales'])
            # print("//Index ", str(layer['index']))
            # print("// Name ", layer['name'])
            # print("// Shape ", layer['shape'])
            # print("// Quantization ", layer['quantization_parameters'])
            # print("// Tensor ")
            # s = np.array2string(interpreter.get_tensor(layer['index']), 80, separator=',')
            # s = s.replace('[', '{')
            # s = s.replace(']', '}')
            # print(s + ';')
            index += 1
        for layer_idx in op['outputs']:
            layer = interpreter._get_tensor_details(layer_idx)
            scales_out = layer['quantization_parameters']['scales']
            print('Output[0] ', layer['name'], layer['quantization_parameters']['scales'])

        print("@@@@ ", np.divide(scales_in, scales_out))

    sys.exit(0)

if True:
    f = open(f'{rep}_parameters.h', 'wt')
    if rep == 'int8_t':
        f.write('#include <inttypes.h>\n')
        f.write('typedef struct int32_8_t {\n  uint32_t mult;\n  uint8_t shift;\n} int32_8_t;\n')

    ops = interpreter._get_ops_details()
    for op in ops:
        t = ""
        # For some reason needed with tf 2.15.0 to avoid duplication
        if op['op_name'] == 'SOFTMAX' :
            break
        for input_idx, layer_idx in enumerate(op['inputs']):
            layer = interpreter._get_tensor_details(layer_idx,0)
            # Seems to me that some 'layers' are intermediate results not needed yet
            # Legacy for float
            if rep == 'float':
                if not re.match(r'sequential', layer['name']) or ';' in layer['name'] or 'Pool' in layer['name']:
                    continue

            wtype = ''
            layer_name = layer["name"].split('/')
            # This matches the naming convention of my lenet C file
            if len(layer_name) > 1:
                if layer_name[2] == 'Conv2D':
                    wtype = 'kernels'
                elif layer_name[2] == 'BiasAdd':
                    wtype = 'biases'
                elif layer_name[2] == 'MatMul':
                    wtype = 'weights'

            if rep == 'int8_t' and (op['op_name'] == 'CONV_2D' or op['op_name'] == 'FULLY_CONNECTED'):
                t += f"/* Quantization parameters: Input[{input_idx}]\n  " + str(layer['quantization_parameters']) + "\n */\n"
                if input_idx == 0: # Zero point to be applied on input data
                    u = f"_zero_points_in[{len(layer['quantization_parameters']['zero_points'])}]" + " = {\n  "
                    for zp in layer['quantization_parameters']['zero_points']:
                        u += str(zp) + ','
                    u += '\n};\n'
                elif input_idx == 1: # Weights, they have the proper name to dump the zero points
                    t += f'int8_t {layer_name[1]}' + u
                elif input_idx == 2: # Bias, as it contains the product of the input scales
                    scales_in = layer['quantization_parameters']['scales']

            if wtype != '':
                layer_shape = re.sub(' +', '][', re.sub('\[ +', '[', f'{layer["shape"]}'))
                if rep == 'int8_t' and layer_name[2] == 'BiasAdd':
                    t += f'int16_t {layer_name[1]}_{wtype}' + f'{layer_shape} =\n'
                else:
                    t += f'{rep} {layer_name[1]}_{wtype}' + f'{layer_shape} =\n'
                s = np.array2string(interpreter.get_tensor(layer_idx), 80, separator=',')
                s = s.replace('[', '{')
                s = s.replace(']', '}')
                t += s + ';\n'

        if rep == 'int8_t' and (op['op_name'] == 'CONV_2D' or op['op_name'] == 'FULLY_CONNECTED'):
            layer = interpreter._get_tensor_details(op['outputs'], 0)
            t += "/* Quantization parameters: Output[0]\n  " + str(layer['quantization_parameters']) + "\n */\n"
            # Dump scale as input_scale0 * input_scale1 / output_scale
            scales_out = layer['quantization_parameters']['scales']
            scales_out = np.divide(scales_in, scales_out)
            t += f'int32_8_t {layer_name[1]}_m0_s[{len(scales_out)}]' + '= {\n'
            for scale in scales_out:
                t += '  {' + str(mult_to_fix_shift(scale)).strip('()') + '},'
            t += '\n};\n'
            # Dump output zero points
            t += f"int8_t {layer_name[1]}_zero_points_out[{len(layer['quantization_parameters']['zero_points'])}]" + " = {\n  "
            for zp in layer['quantization_parameters']['zero_points']:
                t += str(zp) + ','
            t += '\n};\n'
        f.write(t)
    f.close()

# From now on python code to ensure the model behaves as expected
# Change this to test a different image
test_image_index = 2

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file), experimental_preserve_all_tensors = True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]

        predictions[i] = output.argmax()

    return predictions

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index):
    global test_labels

    predictions = run_tflite_model(tflite_file, [test_image_index])

    if False:
        plt.imshow(test_images[test_image_index])
        template = rep + " Model \n True:{true}, Predicted:{predict}"
        _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
        plt.grid(False)
    else:
        print(f"vrai: {test_labels[test_image_index]}, predit: {predictions}\n")


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file):
    global test_images
    global test_labels

    test_image_indices = range(test_images.shape[0])
    predictions = run_tflite_model(tflite_file, test_image_indices)

    accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)

    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (rep, accuracy, len(test_images)))


if False:
    test_model(f"lenet5_models/model_{rep}.tflite", test_image_index)
    all_layers_details = interpreter.get_tensor_details()
    for layer in all_layers_details:
        print("//Index ", str(layer['index']))
        print("// Name ", layer['name'])
        print("// Shape ", layer['shape'])
        print("// Quantization ", layer['quantization_parameters'])
        print("// Tensor ")
        s = np.array2string(interpreter.get_tensor(layer['index']), 80, separator=',')
        s = s.replace('[', '{')
        s = s.replace(']', '}')
        print(s + ';')

if False:
    evaluate_model(f"lenet5_models/model_{rep}.tflite")
