from utils.Feature_Extractor import FeatureExtractor
from classifiers.Logistic_Regression_Classifier import LogisticRegressionClassifier
from Naive_Bayes_Classifier import NaiveBayesClassifier
from utils.util import DataSplitter


class Evaluator:
    def calculate_accuracy(self, predicted_labels, true_labels):
        # Calculate the accuracy of the predictions
        # Return the accuracy
        pass

    def evaluate_hyperparameters(self, data, labels, feature_extraction_methods, smoothing_values, learning_rates,
                                 num_iterations):
        accuracies = []

        for method in feature_extraction_methods:
            # Extract features based on the method
            extractor = FeatureExtractor(method)
            features = extractor.extract_features(data)

            for smoothing in smoothing_values:
                # Split data into training and testing sets
                split = DataSplitter()
                train_data, train_labels, test_data, test_labels = split.split_data(features, labels)

                # Train and test using Naive Bayes
                nb_classifier = NaiveBayesClassifier(smoothing)
                model_nb = nb_classifier.train(train_data, train_labels)
                predicted_labels_nb = nb_classifier.predict(model_nb, test_data)
                accuracy_nb = self.calculate_accuracy(predicted_labels_nb, test_labels)
                accuracies.append(accuracy_nb)

                for learning_rate in learning_rates:
                    # Train and test using Logistic Regression
                    lr_classifier = LogisticRegressionClassifier(learning_rate, num_iterations)
                    model_lr = lr_classifier.train(train_data, train_labels)
                    predicted_labels_lr = lr_classifier.predict(model_lr, test_data)
                    accuracy_lr = self.calculate_accuracy(predicted_labels_lr, test_labels)
                    accuracies.append(accuracy_lr)

        return accuracies
