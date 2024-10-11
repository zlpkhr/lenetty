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

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import sys
import pathlib

# assert float(tf.__version__[:3]) >= 2.3

# récupération du data-set: i == image, l == label (catégorie)
(train_i, train_l), (test_i, test_l) = keras.datasets.mnist.load_data()

# on passe de 28x28 en 32x32 en ajoutant 2 pixels noir tout autour
# le pad étend dans toutes les dimensions, on efface donc les 2 premières
# et 2 dernières images qui sont noires
# on ajoute l'info que c'est un unique canal d'entrée
train_i = np.pad(array = train_i, pad_width = 2, mode = 'constant', constant_values = -1)
train_i = train_i[2:train_i.shape[0] - 2].astype(np.float32)
train_i = np.expand_dims(train_i, axis=-1)
test_i  = np.pad(array = test_i, pad_width = 2, mode = 'constant', constant_values = 0)
test_i  = np.expand_dims(test_i, axis=-1)
test_i  = test_i[2:test_i.shape[0] - 2].astype(np.float32)

val_i = train_i[:5000]
val_l = train_l[:5000]

# lenet de base avec des relu comme fonction d'activation et un softmax à la fin
lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(6, name="C1", kernel_size=5, strides=1,  activation='relu', input_shape=train_i[0].shape, padding='valid'), #C1
    keras.layers.MaxPooling2D(name="S2", pool_size=(2, 2), strides=(2, 2), padding='valid'), #S2
    keras.layers.Conv2D(16, name="C3", kernel_size=5, strides=1, activation='relu', padding='valid'), #C3
    keras.layers.MaxPooling2D(name="S4", pool_size=(2, 2), strides=(2, 2), padding='valid'), #S4
    keras.layers.Flatten(name="F"), #Flatten
    keras.layers.Dense(120, name="F5", activation='relu'), #F5
    keras.layers.Dense(84, name="F6", activation='relu'), #F6
    keras.layers.Dense(10, name="F7", activation='softmax') #F7
])

lenet_5_model.summary()

lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

root_logdir = os.path.join(os.curdir, "log")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

lenet_5_model.fit(train_i, train_l, epochs=5, validation_data=(val_i, val_l), callbacks=[tensorboard_cb])
lenet_5_model.evaluate(test_i, test_l)

# post-training quantization: ou comment passer de float en int8
# doc sur https://www.tensorflow.org/lite/performance/post_training_integer_quant
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_i).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(lenet_5_model)
# Get float model for later (comparison, mainly)
tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)

models_dir = pathlib.Path("./lenet5_models/")
models_dir.mkdir(exist_ok=True, parents=True)

# Save keras stuff (only way to fetch intermediate values I believe)
keras_model_file = models_dir/"model.h5"
lenet_5_model.save(keras_model_file)

# Save float stuff
tflite_model_file = models_dir/"model_float.tflite"
tflite_model_file.write_bytes(tflite_model)

# Save integer stuff
tflite_model_quant_file = models_dir/"model_int8_t.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
