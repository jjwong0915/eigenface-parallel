import numpy


class Eigenface:
    def __init__(self, image_matrix):
        self.image_matrix = image_matrix
        print(self.image_matrix)

    def execute_reduce(self):
        self.mean_image = numpy.mean(self.training_data, axis=1)
        print(self.mean_image)
        return self.mean_image

    def execute_subtract(self):
        self.normalized_matrix = numpy.subtract()
