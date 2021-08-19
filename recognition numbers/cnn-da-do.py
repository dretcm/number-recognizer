from PIL import Image
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
import numpy as np
import numpy
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

try:
    model = load_model('cnn_da_do_model_numbers.h5')
    classes = np.load('data.npy')   
except:

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    classes = np.array(label_binarizer.classes_)
    np.save('data.npy', classes)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)

    rango_rotacion = 30
    mov_ancho = 0.25
    mov_alto = 0.25
    rango_acercamiento=[0.5,1.5]

    datagen = ImageDataGenerator(
                    rotation_range = rango_rotacion,
                    width_shift_range = mov_ancho,
                    height_shift_range = mov_alto,
                    zoom_range=rango_acercamiento,
                    )

    datagen.fit(X_train)

    data_train = datagen.flow(X_train, y_train, batch_size=32)

    model.fit(data_train, validation_data=(X_valid, y_valid), epochs=60, batch_size=128)

    model.save('cnn_da_do_model_numbers.h5')
    model.summary()


def image_to_data(path):
    img = Image.open(path).resize((28,28))  
    new_img = img.convert('L')  # RGB => gray : (R + G + B)/3

    arr = np.array(new_img).astype('float32')/255.0  # (28,28)
    arr = np.expand_dims(arr, axis=0) # (1,28,28)
    arr = np.expand_dims(arr, axis=3) # (1,28,28,1)
    return arr

numbers = [f'n{n}.png' for n in range(10)] # images

def evaluate(numbers):
    fig = plt.figure(figsize=(10,6))
    for i,img in enumerate(numbers):
        ax = plt.subplot(5,4,i*2+1)

        test_img = image_to_data(img)
        ax.imshow(Image.open(img))

        pred = model.predict(test_img).flatten()
        num = classes[numpy.argsort(pred)][-1]

        ax2 = plt.subplot(5,4,i*2+2)
        ax2.bar(classes, pred)
        ax2.set_xticks(classes)
        ax2.set_title('label: '+str(i)+' predict: '+str(num))
        plt.grid()

    plt.tight_layout(h_pad=0, w_pad=0)
    plt.show()
    fig.savefig('cnn-Da-Do-predictions.png')

evaluate(numbers)
