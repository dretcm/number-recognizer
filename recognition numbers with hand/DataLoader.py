import sys, os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

get_labels = lambda files: np.array(list(map(lambda file: int(file[0]), files)))

def get_data(path):
        files = os.listdir(path)
        random.shuffle(files)
        return np.array([cv2.imread(os.path.join(path,file), 0) for file in files], dtype=np.float32), get_labels(files)

def load_data():
        x_train, y_train = get_data("data\data_train")
        x_val, y_val = get_data("data\data_val")
        x_test, y_test = get_data("data\data_test")

        return x_train, y_train, x_val, y_val, x_test, y_test

class OneHotEncoding():
        def __init__(self, labels):
                clases = np.array([], dtype=np.int8)
                for l in labels:
                        if not np.any(clases == l):
                                clases = np.append(clases, l, axis=None)
                self.clases = clases
                self.size = clases.shape[0]
                
        def encoding(self, labels):
                one_hot_clases = np.zeros((labels.shape[0], self.size))
                
                for i in range(labels.shape[0]):
                        one_hot_clases[i][labels[i]] = 1

                return one_hot_clases
        
        def encodings(self, *args):
                out = []
                for arg in args:
                        out.append(self.encoding(arg))
                return out
        
        def get_clases(self):
                return np.sort(self.clases)

def visualizate_data(data, labels):
        for i in range(labels.shape[0]):
                plt.imshow(data[i])
                plt.title(str(labels[i]))
                plt.show()
                
if __name__ == "__main__":
        x_train, y_train, x_val, y_val, x_test, y_test = load_data()
        print(x_train.shape, len(y_train), x_train.dtype)
        print(x_val.shape, len(y_val))
        print(x_test.shape, len(y_test))

##        visualizate_data(x_train, y_train)

##        encoder = OneHotEncoding(y_test)
##        y_train, y_val, y_test = encoder.encodings(y_train, y_val, y_test)
##        clases = encoder.get_clases()
##        print(clases)
##        print(clases.shape)

        

