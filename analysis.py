import tensorflow as tf
import numpy as np
import pathlib

# Load the test dataset
# récupération du data-set: i == image, l == label (catégorie)
(train_i, train_l), (test_i, test_l) = tf.keras.datasets.mnist.load_data()

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
# Normalize the test images to the same range as used during training

# Load the unquantized .h5 model
tflite_model_path = pathlib.Path("./lenet5_models/model_int8_t.tflite")
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path),experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

# Get input and output details for the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tensor_details = interpreter.get_tensor_details()

a = [
"sequential_1/C1_1/Relu;sequential_1/C1_1/add;sequential_1/C1_1/convolution;1",
"sequential_1/S2_1/MaxPool2d",
"sequential_1/C3_1/Relu;sequential_1/C3_1/add;sequential_1/C3_1/convolution;",
"sequential_1/S4_1/MaxPool2d",
"sequential_1/F_1/Reshape",
"sequential_1/F5_1/MatMul;sequential_1/F5_1/Relu;sequential_1/F5_1/Add",
"sequential_1/F6_1/MatMul;sequential_1/F6_1/Relu;sequential_1/F6_1/Add",
"sequential_1/F7_1/MatMul;sequential_1/F7_1/Add",
]

# Evaluate the model on the test dataset
correct_predictions = 0
total_predictions = len(test_l)

for i in range(total_predictions):
    # Prepare input data (expand dimensions to add batch size dimension)
    input_data = np.expand_dims(test_i[i], axis=0).astype(np.uint8)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    for tensor in tensor_details:
        if tensor['name'] in a:
            weights = interpreter.get_tensor(tensor['index'])
            modified_weights = weights * 0
            interpreter.set_tensor(tensor['index'], modified_weights)

    # Run inference
    interpreter.invoke()
    
    # Get the output and determine the predicted label
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    # Check if the prediction matches the actual label
    if predicted_label == test_l[i]:
        correct_predictions += 1

# Calculate and print accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy of the int8 TFLite model: {accuracy * 100:.2f}%")