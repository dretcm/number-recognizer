from PIL import Image
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

try:
    model = load_model('cnn_model_numbers.h5')
    classes = np.load('data.npy')   
except:

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Debido a que son imágenes en escala de grises, el único canal de color se halla implícito. Sin embargo,
    # Keras espera tensores de 4 dimensiones (incluyendo el batch size), no de 3, por lo que tenemos que expandir las dimensiones
    # de los datos.
    
    X_train = np.expand_dims(X_train, axis=3) # [[[1,2,3,4]]] => [[[[1],[2],[3],[4]]]] : (1,1,4) => (1,1,4,1)
    X_test = np.expand_dims(X_test, axis=3)  # (60 000,28,28) => (60 000,28,28,1)

    label_binarizer = LabelBinarizer() # One hot encoder
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    classes = np.array(label_binarizer.classes_)
    np.save('data.npy', classes)
    
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), filters=64, strides=(2, 2), padding='same', activation='relu', kernel_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, batch_size=128)

    model.save('cnn_model_numbers.h5')
    model.summary()


def image_to_data(path):
    img = Image.open(path).resize((28,28))  
    new_img = img.convert('L')  # RGB => gray : (R + G + B)/3

    arr = np.array(new_img).astype('float32')/255.0  # (28,28)
    arr = np.expand_dims(arr, axis=0) # (1,28,28)
    arr = np.expand_dims(arr, axis=3) # (1,28,28,1)
    return arr

test_img = image_to_data('my_number.png')

pred = model.predict(test_img).flatten()

pos = np.flip(np.argsort(pred))
outcome = [str(x) for x in classes[pos]]

title = '>'.join(outcome)

plt.plot(classes, pred)
plt.xticks(classes)
plt.title(title)
plt.grid()
plt.show()
