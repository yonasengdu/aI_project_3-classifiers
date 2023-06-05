
from mnist import MNIST
import math
from os.path import expanduser



class NaiveBayesClassifier:
    def __init__(self, smoothing):
        self.smoothing = smoothing

    def separate_by_class(self,features,labels):
        separated = dict()
        for i in range(len(features)):
            vector = features[i]
            class_value = labels[i]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated
    
    def mean(self,column):
        return sum(column)/len(column)
    
    def stdev(self,column):
        mean = self.mean(column)
        return math.sqrt(sum([(x-mean)**2 for x in column])/(len(column)-1))
    
    def summarize_by_class(self,classDic):
        summarize = dict()
        for class_name, varables in classDic.items():
            summarize[class_name] = [(self.mean(column),self.stdev(column),len(column)) for column in zip(*varables)]
        return summarize
    
    def train(self, features, labels):
        classDic = self.separate_by_class(features,labels)
        self.classes = set(labels)
        self.train_summary = self.summarize_by_class(classDic)

    def calc_probability(self ,x,mean,stdev):
        x = (1/(math.sqrt(2*math.pi)*stdev))
        return x*math.exp(-((x-mean)**2)/(2*stdev**2))
        
    def predict(self,feature):
        prdictDic = dict()
        total_rows = sum([self.train_summary[label][0][2] for label in self.train_summary])
        for c in self.classes:
            prdictDic[c] = (self.train_summary[c][0][2])/(total_rows)
            summary = self.train_summary[c]
            for i in len(summary):
                prdictDic[c] *= self.calc_probability(feature[i],summary[i][0],summary[i][1])
        return prdictDic
        

home = expanduser("~") +"/Documents/aI_project_3-classifiers/sample"
mndata = MNIST(home)
# trainImages,trainLabels = mndata.load_training()
images,labels = mndata.load("../mist_data/t10k-images-idx3-ubyte.gz","../mist_data/t10k-labels-idx1-ubyte",)
# print(testImages)
# mndata.gz = True

# print(trainImages)
# nb = NaiveBayesClassifier(1)

# nb.train(images, labels )