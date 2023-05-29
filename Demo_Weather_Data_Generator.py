import random


class DemoWeatherDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate_data(self):
        data = []
        labels = []
        for _ in range(self.num_samples):
            features = {
                'temperature': random.uniform(20, 40),
                'humidity': random.uniform(40, 90),
                'wind': random.uniform(0, 20),
                'outlook': random.choice(['sunny', 'overcast', 'rainy'])
            }
            if features['temperature'] >= 30 and features['humidity'] >= 70:
                labels.append('hot')
            else:
                labels.append('not hot')
            data.append(features)
        return data, labels
