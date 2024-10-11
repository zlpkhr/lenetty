/*
 * Making float-lenet.c a templated C++ file.
 * Now adding support for other FP formats, in particular those of
 * small size
 *
 * 2020-2024 (c) Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
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

#include <iostream>
#include <cfloat>
#include "float_images.h"
#include "float_parameters.h"
#include "floatx.hpp"

using namespace std;

typedef flx::floatx<5,10> fp16;
typedef flx::floatx<8,7> bfloat16;
typedef flx::floatx<5,2> fp8e5m2;
typedef flx::floatx<4,3> fp8e4m3;

/* Dump tensors more or less as tensorflow does */
template<typename T, int channels, int tensor_size>
void dump_tensor(T input[tensor_size][tensor_size][channels])
{
#ifdef DUMP_TENSORS
    cout << "[";
    for (int i = 0; i < tensor_size; i++) {
        if (i != 0)
            cout << " ";
        cout << "[";
        for (int j = 0; j < tensor_size; j++) {
            if (j != 0)
                cout << "  ";
            cout << "[";
            for (int c = 0; c < channels; c++) {
                cout << input[i][j][c];
                if (c != channels - 1)
                    cout << " ";
            }
            cout << "]";
            if (j != tensor_size - 1) {
                cout << "\n";
            }
        }
        printf("]");
        if (i != tensor_size - 1) {
            cout << "\n";
        }
    }
    cout << "]\n";
#endif
}

/* Dump dense */
template<typename T, int tensor_size>
void dump_dense(T input)
{
#ifdef DUMP_TENSORS
    cout << "[";
    for (int i = 0; i < tensor_size; i++) {
        if (i == 0) {
            cout << "[";
        } else if (i % 8 == 0) {
            cout << "\n  ";
        } else {
            cout << " ";
        }
        cout << input[i];

        if (i == tensor_size - 1) {
            cout << "]";
        }
    }
    cout << "]\n";
#endif
}

/*
 * max is already defined for floats and doubles, so the definition clashes
 * when used with fpnums
 */
template<typename T>
static inline T fpmax(T a, T b)
{
    return a > b ? a : b;
}

template<typename T>
static inline T relu(T a)
{
    return a < T(0) ? T(0) : a;
}

/* FIXME: Handle padding as it should */
template<typename T,
         int in_channels, int out_channels, int img_size, int kernel_size>
void conv2d(T input[img_size][img_size][in_channels],
            T kernel[out_channels][kernel_size][kernel_size][in_channels],
            T bias[out_channels],
            T output[img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                    [img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                    [out_channels])
{
    int fm_size = img_size - 2 * (kernel_size / 2) + !(kernel_size & 1);

    for (int o = 0; o < out_channels; o++) {
        for (int k = 0; k < fm_size; k++) {
            for (int l = 0; l < fm_size; l++) {
                T mac = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        for (int i = 0; i < in_channels; i++) {
                            mac += kernel[o][m][n][i] * input[k + m][l + n][i];
                        }
                    }
                }
                output[k][l][o] = relu(mac + bias[o]);
            }
        }
    }
}

template<typename T, int channels, int img_size, int stride_size>
void maxpool(T input[img_size][img_size][channels],
             T output[img_size / stride_size]
                     [img_size / stride_size]
                     [channels])
{
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int k = 0; k < img_size; k += stride_size) {
                T v = -FLT_MAX;
                for (int m = 0; m < stride_size; m++) {
                    for (int n = 0; n < stride_size; n++) {
                        v = fpmax(v, input[j + m][k + n][i]);
                    }
                }
                output[j / stride_size][k / stride_size][i] = v;
            }
        }
    }
}

template<typename T, int channels, int img_size, int stride_size>
void reshape(T input[img_size][img_size][channels],
             T output[(img_size * img_size * channels) / stride_size])
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

template<typename T, int inputs, int outputs>
void dense(T input[inputs],
           T weight[outputs][inputs],
           T bias[outputs],
           T output[outputs])
{
    for (int j = 0; j < outputs; j ++) {
        output[j] = 0;
        for (int i = 0; i < inputs; i ++) {
            output[j] += input[i] * weight[j][i];
        }
        output[j] = relu(output[j] + bias[j]);
    }
}

/*
 * Not so great a way of doing it, but we're doing some quick hack, so
 * let's live with it, ...
 * Somehow this should be generated directly from tensorflow, btw.
 */
#if FPNUM != float \
    && FPNUM != double \
    && FPNUM != fp16 \
    && FPNUM != bfloat16 \
    && FPNUM != fp8e4m3 \
    && FPNUM != fp8e5m2
#warning "FPNUM macro does not have a predefined supported type\n" \
         "Continuing with custom type, crossing fingers !"
#endif
typedef FPNUM fpnum;

/* Converted parameters */
fpnum fp_C1_kernels[6][5][5][1];
fpnum fp_C1_biases[6];
fpnum fp_C3_kernels[16][5][5][6];
fpnum fp_C3_biases[16];
fpnum fp_F5_weights[120][400];
fpnum fp_F5_biases[120];
fpnum fp_F6_weights[84][120];
fpnum fp_F6_biases[84];
fpnum fp_F7_weights[10][84];
fpnum fp_F7_biases[10];
/* Converted image */
fpnum fp_test_mnist[10000][32][32][1];

template<typename T>
void float2fp_images(int s) 
{
    int i, j, n;

    /* Convert float input into fpnum */
    for (n = 0; n < 10000; n++) {
        for (i = 0; i < 31; i++) {
            for (j = 0; j < 31; j++) {
                /* Lets normalize in some way to avoid overflows */
                fp_test_mnist[n][i][j][0] = T(test_mnist[n][i][j][0]/(float)s);
            }
        }
    }
}

template<typename T>
void float2fp_parameters(void) 
{
    int i, j, k, l;

    /* Convert parameters int fpnum : leaning produces ]-1, +1[ real values */
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 5; j++) {
            for (k = 0; k < 5; k++) {
                for (l = 0; l < 1; l++) {
                    fp_C1_kernels[i][j][k][l] = T(C1_kernels[i][j][k][l]);
                }
            }
        }
    }
    for (i = 0; i < 6; i++) {
        fp_C1_biases[i] = T(fp_C1_biases[i]);
    }

    for (i = 0; i < 16; i++) {
        for (j = 0; j < 5; j++) {
            for (k = 0; k < 5; k++) {
                for (l = 0; l < 6; l++) {
                    fp_C3_kernels[i][j][k][l] = T(C3_kernels[i][j][k][l]);
                }
            }
        }
    }
    for (i = 0; i < 6; i++) {
        fp_C3_biases[i] = T(fp_C3_biases[i]);
    }

    for (i = 0; i < 120; i++) {
        for (j = 0; j < 400; j++) {
            fp_F5_weights[i][j] = T(F5_weights[i][j]);
        }
    }
    for (i = 0; i < 120; i++) {
        fp_F5_biases[i] = T(fp_F5_biases[i]);
    }

    for (i = 0; i < 84; i++) {
        for (j = 0; j < 120; j++) {
            fp_F6_weights[i][j] = T(F6_weights[i][j]);
        }
    }
    for (i = 0; i < 84; i++) {
        fp_F6_biases[i] = T(fp_F6_biases[i]);
    }

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 84; j++) {
            fp_F7_weights[i][j] = T(F7_weights[i][j]);
        }
    }
    for (i = 0; i < 10; i++) {
        fp_F7_biases[i] = T(fp_F7_biases[i]);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
        return -1;
    int s = strtol(argv[1], NULL, 0);

    /* Convert all network parameters to the proper type, and image i too */
    float2fp_parameters<fpnum>();
    float2fp_images<fpnum>(s);

    for (int i = 0; i < 10000; i++) {
        /* Input image 32x32, output image 28x28 */
        fpnum c1_out[28][28][6];
        conv2d<fpnum, 1, 6, 32, 5>(fp_test_mnist[i], fp_C1_kernels, fp_C1_biases, c1_out);
#if 0
        dump_tensor<fpnum, 6, 28>(c1_out);
        exit(0);
#endif
        fpnum s2_out[14][14][6];
        maxpool<fpnum, 6, 28, 2>(c1_out, s2_out);
#if 0
        dump_tensor<fpnum, 6, 14>(s2_out);
        exit(0);
#endif
        fpnum c3_out[10][10][16];
        conv2d<fpnum, 6, 16, 14, 5>(s2_out, fp_C3_kernels, fp_C3_biases, c3_out);
#if 0
        dump_tensor<fpnum, 16, 10>(c3_out);
        exit(0);
#endif
        fpnum s4_out[5][5][16];
        maxpool<fpnum, 16, 10, 2>(c3_out, s4_out);
#if 0
        dump_tensor<fpnum, 16, 5>(s4_out);
        exit(0);
#endif
        fpnum r_out[400];
        reshape<fpnum, 16, 5, 1>(s4_out, r_out);
#if 0
        dump_dense<fpnum, 400>(r_out);
        exit(0);
#endif
        fpnum f5_out[120];
        dense<fpnum, 400, 120>(r_out, fp_F5_weights, fp_F5_biases, f5_out);
#if 0
        dump_dense<fpnum, 120>(f5_out);
        exit(0);
#endif
        fpnum f6_out[84];
        dense<fpnum, 120, 84>(f5_out, fp_F6_weights, fp_F6_biases, f6_out);
#if 0
        dump_dense<fpnum, 84>(f6_out);
        exit(0);
#endif
        fpnum f7_out[10];
        dense<fpnum, 84, 10>(f6_out, fp_F7_weights, fp_F7_biases, f7_out);
#if 0
        dump_dense(10, f7_out);
        exit(0);
#endif

        fpnum v = -FLT_MAX;
        int rank = -1;
        for (size_t i = 0; i < sizeof f7_out/sizeof *f7_out; i++) {
            if (v < f7_out[i]) {
                v = f7_out[i];
                rank = i;
            }
        }
        cout << "got a " << rank << endl;
    }
    return 0;
}
