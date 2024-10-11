/*
 * Raw C implementation of Yann Lecun Lenet.
 * int8_t version that matches TensorFlow Lite outputs.
 *
 * 2020-2023 (c) Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
 * SLS Team, TIMA Lab, Grenoble INP/UGA
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include "int8_t_images.h"
#include "int8_t_parameters.h"

/* Dump tensors more or less as tensorflow does */
void dump_tensor(int channels, int tensor_size,
                 int8_t input[tensor_size][tensor_size][channels])
{
#ifdef DUMP_TENSORS
    printf("[");
    for (int i = 0; i < tensor_size; i++) {
        if (i != 0)
            printf(" ");
        printf("[");
        for (int j = 0; j < tensor_size; j++) {
            if (j != 0)
                printf("  ");
            printf("[");
            for (int c = 0; c < channels; c++) {
                printf("%3d", input[i][j][c]);
                if (c != channels - 1)
                    printf(" ");
            }
            printf("]");
            if (j != tensor_size - 1) {
                printf("\n");
            }
        }
        printf("]");
        if (i != tensor_size - 1) {
            printf("\n");
        }
    }
    printf("]\n");
#endif
}

/* Dump dense */
void dump_dense(int tensor_size,
                int8_t input[tensor_size])
{
#ifdef DUMP_TENSORS
    printf("[");
    for (int i = 0; i < tensor_size; i++) {
        if (i == 0) {
            printf("[");
        } else if (i % 8 == 0) {
            printf("\n  ");
        } else {
            printf(" ");
        }
        printf("%5.5g", input[i]);

        if (i == tensor_size - 1) {
            printf("]");
        }
    }
    printf("]\n");
#endif
}


static inline int8_t max(int8_t a, int8_t b)
{
    return a > b ? a : b;
}

/*
 * There are several linear algebra tricks used by TensorFlow Lite when it
 * targets software, in particular it does not use the zero points in
 * the inner-most loop, but for hw implementation, this is both simple and
 * fast, so let's go for it.
 * Note that by construction tflite guaranties for convolutions that the
 * weights tensors have a zero_point of zero.
 */

static inline int32_t tflite_fixmul(int32_t mac, int32_8_t m0)
{
    /* Compute high part of mult with rounding:
     * we need 64 bits, that sucks */
    int64_t mh = (int64_t)mac * (int64_t)m0.mult;
    int32_t rm = mh >= 0 ? (1 << 30) : (1 - (1 << 30));
    rm = (mh + rm) / (1ll << 31);
#if 1
    /* Compute a rounding arithmetic right shift reverse-engineered
     * from tflite sources */
    int32_t m = (1ll << m0.shift) - 1;
    int32_t u = rm & m;
    int32_t t = (m >> 1) + (rm < 0);
    return (rm >> m0.shift) + (u > t); 
#elif 0
    /* sar */
    int32_t s = -(rm < 0);
    return (s ^ rm) >> m0.shift ^ s;
#elif 0
    /* div */
    return rm >= 0 ? rm >> m0.shift : ~(~rm + 1 >> m0.shift) + 1;
#else
    /* flo */
    return (rm + (1ll << (m0.shift - 1))) >> m0.shift;
#endif
}

/* FIXME: Handle padding as it should */
/* C99 makes my day! */
void conv2d(int in_channels, int out_channels,
            int img_size, int kernel_size,
            int8_t input[img_size][img_size][in_channels],
            int8_t input_zp,
            int8_t kernel[out_channels][kernel_size][kernel_size][in_channels],
            int16_t bias[out_channels],
            int32_8_t m0_s[out_channels],
            int8_t output[img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [out_channels],
            int8_t output_zp)
{
    int fm_size = img_size - 2 * (kernel_size / 2) + !(kernel_size & 1);

    for (int o = 0; o < out_channels; o++) {
        for (int k = 0; k < fm_size; k++) {
            for (int l = 0; l < fm_size; l++) {
                int32_t mac = bias[o];
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        for (int i = 0; i < in_channels; i++) {
                            mac += kernel[o][m][n][i]
                                    * (input[k + m][l + n][i] - input_zp);
                        }
                    }
                }
                mac = tflite_fixmul(mac, m0_s[o]);
                mac += output_zp;
                /* Saturate result */
                mac = mac < -128 ? -128 : mac;
                mac = mac > 127 ? 127 : mac;
                output[k][l][o] = mac;
            }
        }
    }
}

void maxpool(int channels,
             int img_size,
             int stride_size,
             int8_t input[img_size][img_size][channels],
             int8_t output[img_size / stride_size]
                          [img_size / stride_size]
                          [channels])
{
    for (int j = 0; j < img_size; j += stride_size) {
        for (int k = 0; k < img_size; k += stride_size) {
            for (int i = 0; i < channels; i++) {
                int8_t v = SCHAR_MIN;
                for (int m = 0; m < stride_size; m++) {
                    for (int n = 0; n < stride_size; n++) {
                        v = max(v, input[j + m][k + n][i]);
                    }
                }
#if 0
                printf("%d, %d, %d: %d\n", i, k/stride_size, j/stride_size, v);
#endif
                output[j / stride_size][k / stride_size][i] = v;
            }
        }
    }
}

void reshape(int channels,
             int img_size,
             int stride_size,
             int8_t input[img_size][img_size][channels],
             int8_t output[(img_size * img_size * channels) / stride_size])
{
    for (int i = 0; i < img_size; i += stride_size) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int c = 0; c < channels; c++) {
                output[c + j * channels + i * channels * img_size] =
                    input[i][j][c];
            }
        }
    }
}

void dense(int inputs,
           int outputs,
           int8_t input[inputs],
           int8_t input_zp,
           int8_t weight[outputs][inputs],
           int16_t bias[outputs],
           int32_8_t m0_s,
           int8_t output[outputs],
           int8_t output_zp)
{
    for (int j = 0; j < outputs; j ++) {
        int32_t mac = bias[j];
        for (int i = 0; i < inputs; i ++) {
            mac += (input[i] - input_zp) * weight[j][i];
        }

        mac = tflite_fixmul(mac, m0_s);
        mac += output_zp;
        /* Saturate result, seems to do what tflite relu does */
        mac = mac < -128 ? -128 : mac;
        mac = mac > 127 ? 127 : mac;
#if 0
        printf("%d: %d\n", j, mac);
#endif
        output[j] = mac;
    }
}

// Function to flip a random bit in a given array
void flip_random_bit(void* array, size_t array_size) {
    size_t byte_index = rand() % array_size;
    int bit_index = rand() % 8;
    unsigned char* byte_ptr = (unsigned char*)array + byte_index;
    *byte_ptr ^= (1 << bit_index);
    // printf("Flipped bit %d in byte %zu\n", bit_index, byte_index);
}

// Function to perform fault injection
void inject_fault() {
    // Array of pointers to the weight arrays
    void* weight_arrays[] = {
        C1_kernels, C3_kernels, F5_weights, F6_weights, F7_weights
    };
    size_t array_sizes[] = {
        sizeof(C1_kernels), sizeof(C3_kernels), sizeof(F5_weights),
        sizeof(F6_weights), sizeof(F7_weights)
    };
    int num_arrays = sizeof(weight_arrays) / sizeof(weight_arrays[0]);

    // Select a random array
    int array_index = rand() % num_arrays;
    
    // Inject fault in the selected array
    flip_random_bit(weight_arrays[array_index], array_sizes[array_index]);
    
    // printf("Injected fault in array %d\n", array_index);
}

int run_inference(int t) {
    int8_t c1_in[32][32][1];
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            c1_in[i][j][0] = test_mnist[t][i][j][0] - 128;
        }
    }
    int8_t c1_out[28][28][6];
    conv2d(1, 6, 32, 5, c1_in, C1_zero_points_in[0], C1_kernels, C1_biases, C1_m0_s, c1_out, C1_zero_points_out[0]);
    
    int8_t s2_out[14][14][6];
    maxpool(6, 28, 2, c1_out, s2_out);
    
    int8_t c3_out[10][10][16];
    conv2d(6, 16, 14, 5, s2_out, C3_zero_points_in[0], C3_kernels, C3_biases, C3_m0_s, c3_out, C3_zero_points_out[0]);
    
    int8_t s4_out[5][5][16];
    maxpool(16, 10, 2, c3_out, s4_out);
    
    int8_t r_out[400];
    reshape(16, 5, 1, s4_out, r_out);
    
    int8_t f5_out[120];
    dense(400, 120, r_out, F5_zero_points_in[0], F5_weights, F5_biases, F5_m0_s[0], f5_out, F5_zero_points_out[0]);
    
    int8_t f6_out[84];
    dense(120, 84, f5_out, F6_zero_points_in[0], F6_weights, F6_biases, F6_m0_s[0], f6_out, F6_zero_points_out[0]);
    
    int8_t f7_out[10];
    dense(84, 10, f6_out, F7_zero_points_in[0], F7_weights, F7_biases, F7_m0_s[0], f7_out, F7_zero_points_out[0]);

    int8_t v = SCHAR_MIN;
    int rank = -1;
    for (int i = 0; i < sizeof(f7_out)/sizeof(*f7_out); i++) {
        if (v <= f7_out[i]) {
            v = f7_out[i];
            rank = i;
        }
    }
    return rank;
}

int main(int argc, char *argv[])
{
    // Seed the random number generator
    srand(time(NULL));

    // Check if we have the correct number of arguments
    if (argc != 3) {
        return -1;
    }

    // Parse arguments
    int t = strtol(argv[1], NULL, 0);
    int num_flips = (argc == 3) ? strtol(argv[2], NULL, 0) : 0;

    // Run inference without fault injection
    int original_result = run_inference(t);

    // Inject faults if specified
    if (num_flips > 0) {
        for (int i = 0; i < num_flips; i++) {
            inject_fault();
        }
        int flipped_result = run_inference(t);
        printf("%d\n", flipped_result);
    } else {
        printf("%d\n", original_result);
    }

    return 0;
}