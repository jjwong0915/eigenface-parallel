import numpy
import pathlib
import serial
from PIL import Image

EIGENVECTOR_CNT = 4


def load_dataset():
    image_list = []
    label_list = []
    for image_path in pathlib.Path("dataset/yalefaces").glob("subject*"):
        image = Image.open(image_path)
        image_list.append(numpy.asarray(image, dtype=numpy.float32).flatten())
        label_list.append(image_path.name.split(".")[0])
    return numpy.array(image_list), label_list


def main():
    image_data, image_label = load_dataset()
    model = serial.Trainer(eigenvector_cnt=EIGENVECTOR_CNT).train(image_data)
    confidence_list = serial.Predictor(model).predict(image_data)
    print(confidence_list)


if __name__ == "__main__":
    main()
