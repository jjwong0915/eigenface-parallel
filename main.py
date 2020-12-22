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
    image_matrix, image_label = load_dataset()
    eigenface = serial.Eigenface()
    mean_image, weight_vector = eigenface.train(
        train_image=image_matrix.transpose(),
        eigenvector_cnt=EIGENVECTOR_CNT,
    )
    print(mean_image)
    print(weight_vector)
    confidence = eigenface.predict(
        test_image=image_matrix.transpose(),
        mean_image=mean_image,
        weight_vector=weight_vector,
    )


if __name__ == "__main__":
    main()
