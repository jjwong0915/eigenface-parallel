import numpy
import pathlib
import serial
import time
from PIL import Image

EIGENVECTOR_CNT = 4
# IMAGE_PATH = "yalefaces/subject*"
IMAGE_PATH = "ourdatabase/*/*.Jpg"


def load_image(pattern):
    image_list = []
    for image_path in pathlib.Path("dataset").glob(pattern):
        image = Image.open(image_path)
        image_list.append(numpy.asarray(image, dtype=numpy.float32).flatten())
    return numpy.array(image_list)


def evaluate_serial(dataset):
    # train
    trainer = serial.Trainer(eigenvector_cnt=EIGENVECTOR_CNT)
    train_start = time.perf_counter()
    model = trainer.train(dataset)
    train_end = time.perf_counter()
    # predict
    preditor = serial.Predictor(model=model)
    predict_start = time.perf_counter()
    result = preditor.predict(dataset)
    predict_end = time.perf_counter()
    return result, [train_end - train_start, predict_end - predict_start]


def main():
    image_data = load_image(IMAGE_PATH)
    serial_result, serial_time = evaluate_serial(image_data)
    print(serial_time)


if __name__ == "__main__":
    main()
