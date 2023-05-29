from BBC_Data_Loader import BBCDataLoader
from Demo_Weather_Data_Generator import DemoWeatherDataGenerator
from Evaluator import Evaluator
from MNIST_Data_Loader import MNISTDataLoader
from util import AccuracyPlotter


def main():
    # Initialize data loaders
    mnist_loader = MNISTDataLoader('path_to_mnist_dataset_directory')
    bbc_loader = BBCDataLoader('path_to_bbc_dataset_file')
    demo_weather_generator = DemoWeatherDataGenerator(num_samples=100)

    # Load the datasets
    train_images, train_labels, test_images, test_labels = mnist_loader.load_data()
    bbc_data, bbc_labels = bbc_loader.load_data()
    demo_weather_data, demo_weather_labels = demo_weather_generator.generate_data()

    # Define the hyperparameters and feature extraction methods
    smoothing_values = [0.1, 0.5, 1.0, 10, 100]
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
    num_iterations = 100

    # Evaluate hyperparameters and plot accuracy change for MNIST dataset
    evaluator = Evaluator()
    accuracies_mnist = evaluator.evaluate_hyperparameters(train_images, train_labels, ['method1', 'method2', 'method3'],
                                                         smoothing_values, learning_rates, num_iterations)
    plotter = AccuracyPlotter()
    plotter.plot_accuracy_change(accuracies_mnist)

    # Evaluate hyperparameters and plot accuracy change for BBC dataset
    accuracies_bbc = evaluator.evaluate_hyperparameters(bbc_data, bbc_labels, ['method1', 'method2', 'method3'],
                                                       smoothing_values, learning_rates, num_iterations)
    plotter.plot_accuracy_change(accuracies_bbc)

    # Evaluate hyperparameters and plot accuracy change for Demo Weather dataset
    accuracies_demo_weather = evaluator.evaluate_hyperparameters(demo_weather_data, demo_weather_labels,
                                                                 ['method1', 'method2', 'method3'],
                                                                 smoothing_values, learning_rates, num_iterations)
    plotter.plot_accuracy_change(accuracies_demo_weather)


if __name__ == "__main__":
    main()
