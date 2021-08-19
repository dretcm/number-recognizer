from DataLoader import load_data, OneHotEncoding
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class Model():
        def __init__(self, name_model="number_recognizer.h5", epochs=300, name_clases='data_clases.npy'):
                
                self.name_model = name_model
                self.name_clases = name_clases

                self.data = True

                try:
                        self.model = tf.keras.models.load_model(name_model)
                        self.clases = np.load(name_clases)  
                        print("model loaded")
                        
                except Exception as e:
                        print("\n"*10)
                        print(str(e))
                        print("\n"*10)
                        
                        self.init_data()
                        self.epochs = epochs
                        self.init_data_augmentation()
                        self.create_model()
                        self.fit_model()
                        self.save_model()
                        self.draw_history()

        def init_data(self):
                x_train, y_train, x_val, y_val, x_test, y_test = load_data()
                
                self.x_train, self.x_val, self.x_test = map(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], 1),(x_train, x_val, x_test))
                
                self.x_train, self.x_val, self.x_test = map(lambda x: x/255.0, (self.x_train, self.x_val, self.x_test))

                encoder = OneHotEncoding(y_test)
                self.y_train, self.y_val, self.y_test = encoder.encodings(y_train, y_val, y_test)

                self.clases = encoder.get_clases()

                self.height, self.width = x_train.shape[1:]
                self.channels = 1
                self.output = self.clases.shape[0]

                self.data = False

        def init_data_augmentation(self):
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

                datagen.fit(self.x_train)

                self.data_train = datagen.flow(self.x_train, self.y_train, batch_size=32)

        def create_model(self, optimizer="adam", loss="categorical_crossentropy", metric="accuracy"):
                
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.height, self.width, self.channels)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    
                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2,2),
                    
                    tf.keras.layers.Dropout(0.5),
                    
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    
                    tf.keras.layers.Dense(64, activation='relu'),
                    
                    tf.keras.layers.Dense(self.output, activation="softmax")
                    ])

                self.model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=[metric])
                
        def fit_model(self, batch=32, verbose=1):
                print("training the model ...")
                self.history = self.model.fit(
                            self.data_train,
                            epochs = self.epochs,
                            batch_size = batch,
                            validation_data=(self.x_val, self.y_val),
                            verbose=verbose)
                print("finished the training.")
                        
        def evaluate_model(self, batch=32):
                if self.data:
                        self.init_data()
                print(self.model.evaluate(self.x_test, self.y_test, batch_size=batch))

        def predict(self, x):
                x = np.expand_dims(x, axis=[0,3]).astype(np.float32) / 255.0  # (64,64) => (1, 64, 64, 1)

                return self.clases[np.argmax(self.model.predict(x))]
                

        def save_model(self):
                self.model.save(self.name_model)
                np.save(self.name_clases, self.clases)

        def draw_history(self):
                plt.plot(self.history.history["loss"], label="train", c="blue")
                plt.plot(self.history.history["val_loss"], label="validation", c="orange")
                plt.title("History Loss")
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.legend()
                plt.show()

                plt.plot(self.history.history["accuracy"], label="train", c="blue")
                plt.plot(self.history.history["val_accuracy"], label="validation", c="orange")
                plt.title("History Accuracy")
                plt.xlabel("epochs")
                plt.ylabel("accuracy")
                plt.legend()
                plt.show()         

if __name__ == "__main__":
        model = Model()
        img = cv2.imread("data/data_test/5_IMG_5447.jpg", 0)
        plt.imshow(img)
        plt.title(str(model.predict(img)))
        plt.show()
        
        model.evaluate_model()

