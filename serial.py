import numpy


class Model:
    def __init__(self, mean_image, transposed_eigenvector, weight_vector):
        self.mean_image = mean_image
        self.transposed_eigenvector = transposed_eigenvector
        self.weight_vector = weight_vector


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
        self.execute_projection()
        print(self.real_eigenvector)
        return Model(
            mean_image=self.mean_image,
            transposed_eigenvector=self.transposed_eigenvector,
            weight_vector=self.weight_vector,
        )

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

    def execute_projection(self):
        self.weight_vector = numpy.matmul(
            self.transposed_eigenvector,
            self.normalized_image,
        )


class Predictor:
    def __init__(self, model):
        self.mean_image = model.mean_image
        self.transposed_eigenvector = model.transposed_eigenvector
        self.train_weightvector = model.weight_vector

    def predict(self, test_image):
        self.original_image = test_image.transpose()
        self.execute_subtract()
        self.execute_projection()
        self.execute_euclidian()
        return self.confidence_list

    def execute_subtract(self):
        self.normalized_image = self.original_image - self.mean_image

    def execute_projection(self):
        self.test_weightvector = numpy.matmul(
            self.transposed_eigenvector,
            self.normalized_image,
        )

    def execute_euclidian(self):
        self.confidence_list = numpy.array(
            [
                [
                    numpy.linalg.norm(test_weight - train_weight)
                    for train_weight in self.train_weightvector.transpose()
                ]
                for test_weight in self.test_weightvector.transpose()
            ]
        )
