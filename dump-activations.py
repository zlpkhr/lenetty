#!/usr/bin/env python

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

# This utility dumps the activations from the intermediate layers of the NN
# It is just a quick hack to check that my C implementation of Lenet is
# correct, as is my interpretation of the weight and biases storage in the
# H5 model
# Uncomment the layer for which an output is expected

from tensorflow import keras
from keras import backend as kbe
from keras.models import load_model
import numpy as np
import sys

(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
test_images = np.pad(array = test_images, pad_width = 2, mode = 'constant', constant_values = 0)

# Fetch a single image for now
test_images = test_images[2:3]

lenet = load_model('lenet5_models/model.h5')

digit = np.expand_dims(test_images, axis=-1).astype('float32')

# Check the layers one by one to compare with the C implementation
# Dump everything we have
np.set_printoptions(threshold = np.inf, formatter={'float': '{: 0.4f}'.format}, suppress = True)

# c1 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[0].output])
# c1_output = c1([digit])[0]
# print(lenet.layers[0])
# print(c1_output)
# 
# s2 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[1].output])
# s2_output = s2([digit])[0]
# print(lenet.layers[1])
# print(s2_output)
# 
# c3 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[2].output])
# c3_output = c3([digit])[0]
# print(lenet.layers[2])
# print(c3_output)
# 
# s4 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[3].output])
# s4_output = s4([digit])[0]
# print(lenet.layers[3])
# print(s4_output)
#
# r = kbe.function([lenet.layers[0].input],
#                  [lenet.layers[4].output])
# r_output = r([digit])[0]
# print(lenet.layers[4])
# print(r_output)
#
# f5 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[5].output])
# f5_output = f5([digit])[0]
# print(lenet.layers[5])
# print(f5_output)
# 
# f6 = kbe.function([lenet.layers[0].input],
#                   [lenet.layers[6].output])
# f6_output = f6([digit])[0]
# print(lenet.layers[6])
# print(f6_output)
# 
f7 = kbe.function([lenet.layers[0].input],
                  [lenet.layers[7].output])
f7_output = f7([digit])[0]
print(lenet.layers[7])
print(f7_output)

sys.exit(1)

# C1 is okay now (after 3 days of struggling)
# S2 is okay
s2 = lenet.get_layer(name="S2")
extract_s2 = keras.Model(
    inputs=lenet.inputs,
    outputs=s2.output,
)
s2_features = extract_s2(digit)

c3 = lenet.get_layer(name="C3")
extract_c3 = keras.Model(
    inputs=lenet.inputs,
    outputs=c3.output,
)
c3_features = extract_c3(digit)

print(s2_features)
print(c3.input)
print(c3_features)
