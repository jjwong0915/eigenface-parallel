import numpy


class Model:
    def __init__(self, mean_image, weight_vector):
        self.mean_image = mean_image
        self.weight_vector = weight_vector

    def predict(self, test_image):
        pass


class Trainer:
    def __init__(self, eigenvector_cnt):
        self.eigenvector_cnt = eigenvector_cnt

    def train(self, train_image):
        self.original_image = train_image.transpose()
        self.execute_reduce()
        self.execute_subtract()
        self.execute_transpose1()
        self.execute_matmul1()
        self.execute_eigen()
        self.execute_matmul2()
        self.execute_transpose2()
        self.execute_projection1()
        return Model(self.mean_image, self.weight_vector)

    def execute_reduce(self):
        self.mean_image = numpy.mean(
            self.original_image,
            axis=1,
            keepdims=True,
        )

    def execute_subtract(self):
        self.normalized_image = self.original_image - self.mean_image

    def execute_transpose1(self):
        self.transposed_image = self.normalized_image.transpose()

    def execute_matmul1(self):
        self.covariance_matrix = numpy.matmul(
            self.transposed_image,
            self.normalized_image,
        )

    def execute_eigen(self):
        values, vectors = numpy.linalg.eig(self.covariance_matrix)
        self.temp_eigenvector = vectors[
            numpy.argsort(values)[-self.eigenvector_cnt :]
        ].transpose()

    def execute_matmul2(self):
        self.real_eigenvector = numpy.matmul(
            self.normalized_image,
            self.temp_eigenvector,
        )

    def execute_transpose2(self):
        self.transposed_eigenvector = self.real_eigenvector.transpose()

    def execute_projection1(self):
        self.weight_vector = numpy.matmul(
            self.transposed_eigenvector,
            self.normalized_image,
        )
