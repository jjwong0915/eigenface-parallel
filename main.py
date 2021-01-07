import numpy
import pathlib
import serial
import parallel
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
    return [train_end - train_start, predict_end - predict_start]


def evaluate_parallel(dataset):
    # train
    NO_IMAGES = dataset.shape[0]
    NO_PIXELS = dataset.shape[1]
    V = numpy.empty((NO_IMAGES, NO_IMAGES), dtype=numpy.float32)
    E = numpy.empty((NO_IMAGES, EIGENVECTOR_CNT), dtype=numpy.float32)
    result = numpy.empty((NO_IMAGES, NO_IMAGES), dtype=numpy.float32)
    dataset_T = dataset.transpose()
    train = parallel.eigenface(
        dataset_T,
        NO_IMAGES,
        NO_PIXELS,
        EIGENVECTOR_CNT,
        32)
    train_start = time.perf_counter()
    train.training_step1()
    train.C_fetch(V)
    eigenvalues, eigenvectors = numpy.linalg.eig(V)
    E = eigenvectors[numpy.argsort(eigenvalues)[-EIGENVECTOR_CNT:]]
    train.eigenvector(E)
    train.training_step2()
    train_end = time.perf_counter()
    # predict
    predict_start = time.perf_counter()
    train.testing(dataset_T)
    train.confident_fetch(result)
    predict_end = time.perf_counter()
    return [train_end - train_start, predict_end - predict_start]


def main():
    # change input data size
    image_data = load_image(IMAGE_PATH)[:20]
    print(len(image_data))
    serial_time = evaluate_serial(image_data)
    print(serial_time)
    parallel_time = evaluate_parallel(image_data)
    print(parallel_time)


if __name__ == "__main__":
    main()
