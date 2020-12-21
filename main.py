import numpy
import pathlib
import serial
from PIL import Image

NO_IMAGES = 165
NO_PIXELS = 77760
NO_EIGENS = 4
NO_SUBJECT = 16


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


if __name__ == "__main__":
    main()
