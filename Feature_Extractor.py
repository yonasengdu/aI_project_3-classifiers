class FeatureExtractor:
    def __init__(self, method):
        self.method = method

    def extract_features(self, data):
        if self.method == 'method1':
            return self.extract_method1(data)
        elif self.method == 'method2':
            return self.extract_method2(data)
        elif self.method == 'method3':
            return self.extract_method3(data)
        else:
            raise ValueError('Invalid feature extraction method')

    def extract_method1(self, data):
        # Implement feature extraction method 1
        pass

    def extract_method2(self, data):
        # Implement feature extraction method 2
        pass

    def extract_method3(self, data):
        # Implement feature extraction method 3
        pass

