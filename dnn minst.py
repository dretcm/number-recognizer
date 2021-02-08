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
    model.add(Dense(512, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=200,verbose=2)
    model.save('model_numbers.h5')

    # model.summary()

def image_to_data(path):
    img = Image.open(path).resize((28,28)) # 28x28 pixels
    # img.save('new_im.png')
    new_img = img.convert('L')  # RGB => gray : (R + G + B)/3
    plt.subplot(221)
    plt.imshow(new_img)
    plt.subplot(222)
    plt.imshow(new_img, cmap='gray')
    plt.show()
    arr = numpy.array(new_img.getdata()).reshape((1,784)).astype('float32')/255.0
    return arr

test_img = image_to_data('my_number.png') #changue image with paint

pred = model.predict(test_img).flatten()

pos = numpy.flip(numpy.argsort(pred))
outcome = [str(x) for x in classes[pos]]

title = '>'.join(outcome)

plt.plot(classes, pred)
plt.xticks(classes)
plt.title(title)
plt.grid()
plt.show()

