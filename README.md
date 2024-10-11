# Lenet in C99

This piece of code has been developed to understand how tflite does its computations, in order for example to target hardware implementations.
This work also had the side effect of understanding how the parameters are stored in TensorFlow output files.

Python programs are used to generate the input images (MNIST data-set, not a surprise for Lenet) and dump the either float or int8 parameters.
Note that this has been recently ported to TensorFlow 2.15.0 (adding one argument to `_get_tensors_details` and so expects (at least) this version of TensorFlow to work).

However, for convenience since this python stuff is always a headache to setup, we also provide the generated files for the parameters and input images.


The relevant documentations that helped me to understand are:

[Quantization spec](https://www.tensorflow.org/lite/performance/quantization_spec), where it is explained that (quote) “Weights are symmetric: forced to have zero-point equal to 0.”

[More on quantization](https://github.com/google/gemmlowp/blob/master/doc/quantization.md), where it is explained that the new scheme is a bit different from what is said in Google’s original paper.

[Reference code](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/conv.h#L101), that does the “whole” computation (i.e. ignores that zero-points is 0 for weights), but otherwise relatively easy to understand.

Note that the `tflite_fixmul` function has been reversed engineered from the code
and does a rounding to nearest with tie to away (thanks to Florent De Dinechin for the explanations).

I included a  simple shift arithmetic right, a usual division, and a rounding high as alternative to the original tflite computation.
Playing with these gives, with “label” the ground truth, “ref” the result of vanilla tensorflow-lite, and “sar”, “div” and “flo” the above mentionned solutions, we have: 
```
diff 1.label 1.ref | grep -- '---' | wc -l => 181, précision => .98190
diff 1.label 1.sar | grep -- '---' | wc -l => 186, précision => .98140
diff 1.label 1.div | grep -- '---' | wc -l => 186,
diff 1.label 1.flo | grep -- '---' | wc -l => 186,

diff 1.ref   1.div | grep -- '---' | wc -l => 16
diff 1.ref   1.sar | grep -- '---' | wc -l => 16
diff 1.ref   1.flo | grep -- '---' | wc -l => 15
diff 1.sar   1.div | grep -- '---' | wc -l => 0
diff 1.sar   1.flo | grep -- '---' | wc -l => 11
diff 1.div   1.flo | grep -- '---' | wc -l => 11
```

# Lenet in C++

Using the floatx.hpp header (`https://github.com/oprecomp/FloatX.git`), we can, given a templated C++ implementation, test with various floating-point representations.

A kind of dirty hack first casts all parameters of the network and the input image into the appropriate `floatx<y,z>`.
Then we just apply the float functions, but templated.
