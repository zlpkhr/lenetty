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

# Building a C array from the original MNIST images 
# Works only for the MNIST data set, but easilly expandable I suppose

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

if len(sys.argv) != 2:
    print(f'Usage {sys.argv[0]} (float|uint8_t|int8_t)')
    sys.exit(1)

if  sys.argv[1] == "uint8_t" or sys.argv[1] == "float" or sys.argv[1] == "int8_t":
    rep = sys.argv[1]
else:
    print(f'Usage {sys.argv[0]} (float|uint8_t|int8_t)')
    sys.exit(1)

(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()

# From 28x28 to 32x32 to avoid handling corner cases during NN evaluation
test_images = np.pad(array = test_images, pad_width = 2, mode = 'constant', constant_values = 0)
# Since this add 2 pictures before and after the figure, just skip them
test_images = test_images[2:10002]
# Small check to make sure we got what we expected
# np.set_printoptions(threshold = np.inf)
# print(test_images)
# sys.exit(0)

# s must be initialized to be used later
s = ''

if rep == 'uint8_t' or rep == 'int8_t':
    s += '#include <inttypes.h>\n'
else:
    test_images = test_images.astype(float)
    # This would normalize between 0 and 1, but we don't want to do that
    # test_images = tf.image.convert_image_dtype(test_images, dtype=tf.float32, saturate=False)

s += f'{rep}' + ' test_mnist[][32][32][1] = {'
for l, j in enumerate(test_images):
    s += '\n{' + f'// expected label: {test_labels[l]}\n'
    for i in j:
        s += '{'
        for v in i:
            if rep == 'int8_t':
                u = "{:d}".format(v)
            elif rep == 'uint8_t':
                u = "0x{0:02x}".format(v)
            else:
                u = "{:f}".format(v)
            s += '{' + f'{u}' + '},'
        s += '}\n,'
    s += '},\n'
s += '};\n'

# Dump the data as a c file for separate compilation
f = open(f'{rep}_images.c', 'wt')
f.write(s)
f.close()
# Dump a header containing the type
s = f'extern {rep}' + ' test_mnist[][32][32][1];'
