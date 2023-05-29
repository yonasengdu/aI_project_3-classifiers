from mnist import MNIST


class MNISTDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        mndata = MNIST(self.dataset_path)
        train_images, train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()
        return train_images, train_labels, test_images, test_labels
