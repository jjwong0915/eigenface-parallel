import numpy
import pathlib
import serial
from PIL import Image

SUBJECT_CNT = 16
EIGENVECTOR_CNT = 4


def load_image():
    image_list = []
    for image_path in pathlib.Path("dataset/yalefaces").glob("subject*"):
        image = Image.open(image_path)
        image_list.append(numpy.asarray(image).flatten())
    return numpy.array(image_list, dtype=numpy.float32).transpose()


def main():
    image_matrix = load_image()
    eigenface = serial.Eigenface(image_matrix)
    mean_image = eigenface.execute_reduce()
    eigenface.execute_subtract()
    eigenface.execute_transpose()
    eigenface.execute_matmul1()
    eigenface.execute_eigen(EIGENVECTOR_CNT)


if __name__ == "__main__":
    main()
