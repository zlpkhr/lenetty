### int-lenet Index

**MNIST = Array of 10,000 test images**  
`INDEX -> MNIST[]`

Examples of usage:  
- `int-lenet 0` → Fetches the first image from the MNIST dataset and passes it to the `int-lenet` program to predict the digit.  
- `int-lenet 14` → Fetches the 15th image from the MNIST dataset and passes it to the `int-lenet` program for digit prediction.

### What Have We Done?

We've modified the `int-lenet` program to accept an additional parameter:

```
int-lenet INDEX NUMBER_OF_INJECTIONS
```

- `INDEX` is the same as before, referring to the image index in the MNIST dataset.
- `NUMBER_OF_INJECTIONS` specifies how many random bit flips to inject into the model parameters.

We inject bit flips randomly across the model's parameters. This is done by selecting random parameters within a layer. The layers and their corresponding parameters are:

- `C1_kernels`
- `C3_kernels`
- `F5_weights`
- `F6_weights`
- `F7_weights`

Each parameter set corresponds to a specific layer (e.g., `C1` corresponds to `C1_kernels`).

We can also work with biases, though this has been omitted in our current implementation.

#### Bit Flipping Process:
1. Randomly choose a layer from the parameters (e.g., `C1_kernels`).
2. Randomly select a value (weight) from the chosen layer.
3. Flip a random bit in this value.

### Example Commands:

- `int-lenet 0 0` → Predicts the digit for the first image with 0 bit flips.
- `int-lenet 0 10` → Predicts the digit for the first image with 10 bit flips injected.
- `int-lenet 122 1000` → Predicts the digit for the 123rd image with 1000 bit flips.

### Why Modify the Same Model Code?

Our approach maintains the original code as much as possible, based on the following workflow:

1. **Model Training:**  
   We train a LeNet-5 model using Python and TensorFlow for convenience.
   
2. **Model Optimization:**  
   The model is saved and optimized using TensorFlow Lite (TFLite), targeting 8-bit integer quantization (int8) for efficient use in embedded systems.
   
3. **C Implementation:**  
   The model is then implemented in C (`int-lenet.c`) because C provides better performance on embedded systems. However, training in C is impractical due to its complexity and low productivity compared to Python.
   
4. **Splitting Training and Inference:**  
   To balance ease of training with performance, we handle training in Python and inference in C. The problem is that TensorFlow Lite's saved model format isn't directly compatible with the C code. Hence, the professor wrote scripts (found in the `lenet5_models` folder) to convert TensorFlow Lite models into C-compatible files (`int8_parameters.h` for weights, etc.).

5. **No Original int8 Evaluation Script:**  
   The provided script (`accuracy.sh`) evaluates only the float model, which is very slow. There was no script to evaluate the int8 model accuracy, so we implemented one for the bit-flipping experiment directly in `int-lenet.c` to streamline the process.

#### Why Modify `int-lenet` for Injections?

If we didn't modify the code, we'd have to manually inject bit flips through Python scripts and repeatedly compile the C code for every fault injection scenario, which would be inefficient. By implementing fault injection directly into `int-lenet`, we only need to compile once and then run as many tests as we like.

### Fault Injection Implementation:

Instead of generating separate models for each bit-flip scenario, we:
1. Train the model.
2. Preprocess and convert the model using `dump-images.py` and `dump-parameters.py`.
3. Compile `int-lenet`.
4. Run `int-lenet` for any number of injections (0 to 10000) across the dataset, varying the number of bit flips.

This avoids repeated recompilation and streamlines the testing process.

### Future Considerations:

- **Layer-Specific Fault Injections:**  
   Ideally, we should allow users to specify which layer to inject faults into, including activation values (intermediate outputs between layers) which aren't implemented yet.

- **Fault Injection in Activations:**  
   Real-world fault injections might also affect the values transferred between neurons, leading to incorrect computations. This should be considered in future implementations.

### Automation:

Manually testing the model is possible, but for efficient performance measurement, automation is needed. Testing bit flips from 0 to 50 across the entire dataset (10,000 images) involves 500,000 computations, which is computationally intensive.

### Parallelization:

To speed up the process, we parallelized the task using Go. The Go script (`collect_fi_results.go`) runs multiple predictions simultaneously:

- Each row corresponds to an image.
- For each image, the program computes the predictions for all levels of bit flips (0 to 50).

The results are saved in a text file, although the format isn't optimal.

### Data Processing:

We wrote a Python script to:
1. Read and preprocess the results from the Go script.
2. Compare predictions against the correct labels.
3. Calculate accuracy for each level of fault injection (0 to 50 bit flips).
4. Display the results as a graph.

### Areas for Improvement:

- **Refine Fault Injection:**  
   Implement targeted bit flips and include activations and biases.
   
- **Bit Flip Direction:**  
   Allow control over whether to flip from 0 → 1 or 1 → 0.

- **Better Result Storage:**  
   Use a more structured file format, like CSV, for storing results.

- **Port to Python:**  
   Consider removing the C model and moving everything to Python for easier development and automation.

- **True Randomization:**  
   Improve randomization to be suitable for research by using a more robust random number generator.

- **Further Automation:**  
   Implement continuous integration (CI) pipelines for fully automated testing.

---

### Commands:

**Cleanup:**
```
make clean
```

**Install dependencies:**
```
pip install tensorflow numpy matplotlib
```

**Train and Save Models:**
```
python lenet.py
```

**Preprocess Models:**
```
python dump-images.py
python dump-parameters.py
```

**Compile Model in C:**
```
make int-lenet
```

**Compile Golang Script:**
```
go build collect_fi_results.go
```

**Run Golang Script:**
```
./collect_fi_results
```