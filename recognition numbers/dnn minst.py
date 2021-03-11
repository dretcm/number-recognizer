import numpy
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
# from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer #OTHER OPTION

from PIL import Image
import matplotlib.pyplot as plt

try:
    model = load_model('model_numbers.h5')
    classes = numpy.load('data.npy')

except:
    numpy.random.seed(10)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    num_pixels = X_train.shape[1] * X_train.shape[2] # 28x28
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # (60000,784)
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

    # normaliza las entradas de (0-255) a (0-1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # y_train = np_utils.to_categorical(y_train) # [1,2,3] => [[1,0,0],[0,1,0],[0,0,1]]
    # y_test = np_utils.to_categorical(y_test)
    # num_classes = y_test.shape[1]

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)
    num_classes = y_test.shape[1]

    classes = numpy.array(label_binarizer.classes_)
    numpy.save('data.npy', classes)

    model = Sequential()
    model.add(Dense(64, input_dim=num_pixels, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)
    model.save('model_numbers.h5')

    # model.summary()

def image_to_data(path):
    img = Image.open(path)
    img = img.convert('L')  # RGB => gray : (R + G + B)/3
    new_img = img.resize((28,28)) # 28x28 pixels
    arr = numpy.asarray(new_img).reshape((1,784)).astype('float32')/255.0
    return arr

numbers = [f'n{n}.png' for n in range(10)] # images

def evaluate(numbers):
    fig = plt.figure(figsize=(10,6))
    for i,img in enumerate(numbers):
        ax = plt.subplot(5,4,i*2+1)

        test_img = image_to_data(img)
        ax.imshow(test_img.reshape((28,28)))

        pred = model.predict(test_img).flatten()
        num = classes[numpy.argsort(pred)][-1]

        ax2 = plt.subplot(5,4,i*2+2)
        ax2.bar(classes, pred)
        ax2.set_xticks(classes)
        ax2.set_title('label: '+str(i)+' predict: '+str(num))
        plt.grid()

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.show()
    fig.savefig('predictions.png')

evaluate(numbers)

# for i in range(10):
#     img = Image.open(numbers[i]).resize((28,28))
#     aux = numpy.asarray(img.convert('L')).astype('float32')/255.0
#     pred = classes[numpy.argsort(model.predict(aux.reshape(1, 784)).flatten())][-1]
#     print(f' label: {classes[i]} - prdict: {pred}')

