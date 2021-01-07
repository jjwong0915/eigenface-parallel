import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import pathlib

from PIL import Image
from math import ceil
from pycuda.compiler import SourceModule

module_reduce = SourceModule("""
__global__ void reduce_kernel(float* mean_vector, float* input_vector)
{
    int interval;
    extern __shared__ float local_vector[];
    local_vector[threadIdx.x] = input_vector[blockIdx.x * \
        blockDim.x + threadIdx.x];
    __syncthreads();
    for (interval = 2; interval <= blockDim.x; interval *= 2) {
        if (threadIdx.x % interval == 0 && threadIdx.x + (interval / 2) < blockDim.x) {
            local_vector[threadIdx.x] += local_vector[threadIdx.x + \
                (interval / 2)];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (interval / 2 != blockDim.x) {
            local_vector[0] += local_vector[interval / 2];
        }
        mean_vector[blockIdx.x] = local_vector[0] / (float)blockDim.x;
    }
}
""")

module_subtract = SourceModule("""
__global__ void subtract_kernel(float* A, float* train, float* average)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    A[bx * blockDim.x + tx] = train[bx * blockDim.x + tx] - average[bx];
}
""")

module_transpose = SourceModule("""
__global__ void transpose_kernel(float* transpose_A, float* A, int NO_PIXELS, int NO_IMAGES)
{
    int Ax = blockIdx.x * blockDim.x + threadIdx.x;
    int Ay = blockIdx.y * blockDim.y + threadIdx.y;
    if (Ax < NO_PIXELS && Ay < NO_IMAGES) {
        transpose_A[Ay * NO_PIXELS + Ax] = A[Ax * NO_IMAGES + Ay];
    }
}
""")

module_matmul = SourceModule("""
__global__ void matmul_kernel(float* A, float* B, float* C, int Ax, int Ay, int By)
{
    const int TILE_WIDTH = 32;
    __shared__ float tileMs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileNs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blx = blockIdx.x;
    int bly = blockIdx.y;
    int ax = Ax, ay = Ay;
    int bx = Ay, by = By;
    int cx = Ax, cy = By;

    // target element coordinates
    int row = bly * TILE_WIDTH + ty;
    int column = blx * TILE_WIDTH + tx;

    float pValue = 0;
    int bound = ceilf(ay / (float)TILE_WIDTH);
    // compute target element value
    for (unsigned int i = 0; i < bound; i++) {
        // move the tiles and update shared memory value for new tile positions
        unsigned int itw = i * TILE_WIDTH;
        if (row < ax && (itw + tx) < ay)
            tileMs[ty][tx] = A[row * ay + itw + tx];
        else
            tileMs[ty][tx] = 0;
        if (column < by && (itw + ty) < bx)
            tileNs[ty][tx] = B[(itw + ty) * by + column];
        else
            tileNs[ty][tx] = 0;

        // after the entire tile's values are available, proceed
        __syncthreads();

        for (unsigned int j = 0; j < TILE_WIDTH; j++)
            pValue += tileMs[ty][j] * tileNs[j][tx];
        // after the entire tile's values have been used, proceed
        __syncthreads();
    }
    // boundary check
    if (row < cx && column < cy)
        C[row * cy + column] = pValue;
}

__global__ void matmul_kernel2(float* A, float* B, float* C, int Ax, int Ay, int By)
{
    const int TILE_WIDTH = 32;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blx = blockIdx.x;
    int bly = blockIdx.y;
    int ax = Ax, ay = Ay;
    int bx = Ay, by = By;
    int cx = Ax, cy = By;

    int row = bly * TILE_WIDTH + ty;
    int column = blx * TILE_WIDTH + tx;

    float pValue = 0;
    int bound = ceilf(ay / (float)TILE_WIDTH);
    for (unsigned int i = 0; i < bound; i++) {
        unsigned int itw = i * TILE_WIDTH;
        for (unsigned int j = 0; j < TILE_WIDTH; j++){
            if(row < ax && (itw + tx) < ay && column < by && (itw + ty) < bx){
                pValue += A[row * ay + itw + j] * B[(itw + j) * by + column];
            }
        }
        // after the entire tile's values have been used, proceed
        __syncthreads();
    }
    // boundary check
    if (row < cx && column < cy)
        C[row * cy + column] = pValue;
}
""")

module_euclidiandist = SourceModule("""
__global__ void euclidiandist_kernel(float* confident, float* train, float* test, int NO_EIGENVALUE)
{
    const int NO_TEST_IMAGES = blockDim.x;
    const int NO_IMAGES = gridDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int i;
    float diff;
    float total_diff = 0.0;
    for(i = 0; i < NO_EIGENVALUE; i++){
        float train_data = train[i * NO_IMAGES + bx];
        float test_data = test[i * NO_TEST_IMAGES + tx];
        diff = train_data - test_data;
        total_diff += diff;
    }
    float conf = 1.0f - sqrtf(total_diff*total_diff/(float)NO_EIGENVALUE*NO_IMAGES)/255.0f;
    confident[bx * NO_TEST_IMAGES + tx] = conf;
}
""")

reduce_kernel = module_reduce.get_function("reduce_kernel")
subtract_kernel = module_subtract.get_function("subtract_kernel")
transpose_kernel = module_transpose.get_function("transpose_kernel")
matmul_kernel = module_matmul.get_function("matmul_kernel2")
euclidiandist_kernel = module_euclidiandist.get_function(
    "euclidiandist_kernel")


class eigenface:
    def __init__(self, image_matrix, NO_IMAGES, NO_PIXELS, NO_EIGENVALUE, TILE_DIM):
        self.NO_IMAGES = NO_IMAGES
        self.NO_PIXELS = NO_PIXELS
        self.NO_EIGENVALUE = NO_EIGENVALUE
        self.TILE_DIM = TILE_DIM
        self.training_ptr = cuda.mem_alloc(image_matrix.nbytes)
        cuda.memcpy_htod(self.training_ptr, image_matrix)
        self.A = cuda.mem_alloc(image_matrix.nbytes)
        self.transpose_A = cuda.mem_alloc(image_matrix.nbytes)
        self.U = cuda.mem_alloc(self.NO_PIXELS * self.NO_EIGENVALUE * 4)
        self.transpose_U = cuda.mem_alloc(
            self.NO_PIXELS * self.NO_EIGENVALUE * 4)
        self.weight_train = cuda.mem_alloc(NO_EIGENVALUE * self.NO_IMAGES * 4)

    def training_step1(self):
        self.average_ptr = cuda.mem_alloc(self.NO_PIXELS * 4)
        # reduce
        self.reduce_kernel()
        # subtract
        self.subtract_kernel()
        # transpose A
        self.transpose_kernel(self.A, self.transpose_A,
                              self.NO_PIXELS, self.NO_IMAGES)
        self.C = cuda.mem_alloc(self.NO_IMAGES * self.NO_IMAGES * 4)
        # matmul tranpose_A and A
        self.matmul_kernel(self.transpose_A, self.A, self.C,
                           self.NO_IMAGES, self.NO_PIXELS, self.NO_IMAGES)
        # wait eigenvector caculate

    def training_step2(self):
        self.matmul_kernel(self.A, self.V, self.U,
                           self.NO_PIXELS, self.NO_IMAGES, self.NO_EIGENVALUE)
        self.transpose_kernel(self.U, self.transpose_U,
                              self.NO_PIXELS, self.NO_EIGENVALUE)
        self.matmul_kernel(self.transpose_U, self.A, self.weight_train,
                           self.NO_EIGENVALUE, self.NO_PIXELS, self.NO_IMAGES)

    def testing(self, test_image):
        self.NO_TEST_IMAGES = test_image.shape[1]
        self.testing_ptr = cuda.mem_alloc(test_image.nbytes)
        cuda.memcpy_htod(self.testing_ptr, test_image)
        self.weight_test = cuda.mem_alloc(
            self.NO_EIGENVALUE * self.NO_TEST_IMAGES * 4)
        self.matmul_kernel(self.transpose_U, self.testing_ptr, self.weight_test,
                           self.NO_EIGENVALUE, self.NO_PIXELS, self.NO_IMAGES)
        self.confident = cuda.mem_alloc(
            self.NO_IMAGES * self.NO_TEST_IMAGES * 4)
        self.euclidiandist_kernel()

    def C_fetch(self, cpu_mem):
        cuda.memcpy_dtoh(cpu_mem, self.C)

    def eigenvector(self, cpu_mem):
        self.V = cuda.mem_alloc(self.NO_IMAGES * self.NO_EIGENVALUE * 4)
        cuda.memcpy_htod(self.V, cpu_mem)

    def reduce_kernel(self):
        reduce_kernel(
            self.average_ptr, self.training_ptr,
            block=(self.NO_IMAGES, 1, 1),
            grid=(self.NO_PIXELS, 1),
            shared=self.NO_IMAGES * 4
        )

    def subtract_kernel(self):
        subtract_kernel(
            self.A, self.training_ptr, self.average_ptr,
            block=(self.NO_IMAGES, 1, 1),
            grid=(self.NO_PIXELS, 1)
        )

    def transpose_kernel(self, V, transpose_V, V_x, V_y):
        transpose_kernel(
            transpose_V, V,
            numpy.int32(V_x),
            numpy.int32(V_y),
            block=(self.TILE_DIM, self.TILE_DIM, 1),
            grid=(ceil(V_x / self.TILE_DIM),
                  ceil(V_y / self.TILE_DIM))
        )

    def matmul_kernel(self, In1, In2, Out, In1_x, In1_y, In2_y):
        matmul_kernel(
            In1, In2, Out,
            numpy.int32(In1_x),
            numpy.int32(In1_y),
            numpy.int32(In2_y),
            block=(self.TILE_DIM, self.TILE_DIM, 1),
            grid=(ceil(In1_x / self.TILE_DIM),
                  ceil(In2_y / self.TILE_DIM)),
            # shared=self.TILE_DIM * self.TILE_DIM * 2 * 4,
        )

    def euclidiandist_kernel(self):
        euclidiandist_kernel(
            self.confident,
            self.weight_train,
            self.weight_test,
            numpy.int32(self.NO_EIGENVALUE),
            block=(self.NO_TEST_IMAGES, 1, 1),
            grid=(self.NO_IMAGES, 1),
        )

    def average_fetch(self, v):
        cuda.memcpy_dtoh(v, self.average_ptr)

    def confident_fetch(self, v):
        cuda.memcpy_dtoh(v, self.confident)
