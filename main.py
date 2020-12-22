import numpy
import pathlib
import serial
from PIL import Image

EIGENVECTOR_CNT = 4


def load_image():
    image_list = []
    for image_path in pathlib.Path("dataset/yalefaces").glob("subject*"):
        image = Image.open(image_path)
        image_list.append(numpy.asarray(image, dtype=numpy.float32).flatten())
    return numpy.array(image_list)


def main():
    image_data = load_image()
    serial_model = serial.Trainer(
        eigenvector_cnt=EIGENVECTOR_CNT,
    ).train(image_data)
    confidence_list = serial.Predictor(serial_model).predict(image_data[100])
    print(confidence_list)


if __name__ == "__main__":
    main()
