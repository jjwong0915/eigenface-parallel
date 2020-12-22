import numpy


class Eigenface:
    def __init__(self, image_matrix):
        self.image_matrix = image_matrix
        print(self.image_matrix)

    def execute_reduce(self):
        self.mean_image = numpy.mean(self.image_matrix, axis=1, keepdims=True)
        print(self.mean_image)
        return self.mean_image

    def execute_subtract(self):
        self.normalized_matrix = self.image_matrix - self.mean_image
        print(self.normalized_matrix)

    def execute_transpose(self):
        self.transposed_matrix = self.normalized_matrix.transpose()
        print(self.transposed_matrix)

    def execute_matmul1(self):
        self.covariance_matrix = numpy.matmul(
            self.transposed_matrix,
            self.normalized_matrix,
        )
        print(self.covariance_matrix)

    def execute_eigen(self, eigenvector_cnt):
        values, vectors = numpy.linalg.eig(self.covariance_matrix)
        self.eigenvector_list = vectors[
            numpy.argsort(values)[-eigenvector_cnt:]
        ]
        print(self.eigenvector_list)

    def execute_matmul2(self):
        pass
