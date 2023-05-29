import csv


class BBCDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        data = []
        labels = []
        with open(self.dataset_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                labels.append(row[0])
                data.append(row[1])
        return data, labels
