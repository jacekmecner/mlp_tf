# import os
#
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from random import random
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

# array([[0.1, 0.2], [0.2, 0.2]])
# array([[0.3]. [0.4])

def generate_dataset(num_samples, test_size):
    # create an array x of shape (2000, 2), where each row contains two random numbers between 0 and 0.5.
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    # y: create 1D numpy array of length 2000 where each value is the sum of the two random numbers from the corresponding row in x.
    y = np.array([i[0] + i[1] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    # print(f"x_test: \n{x_test}")
    # print(f"y_test: \n{y_test}")


    # build model: 2 -> 5 -> 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # compile model
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser, loss="MSE")

    # train model
    model.fit(x_train, y_train, epochs=100)

    # evaluate model
    print("\nModel evaluation:")
    model.evaluate(x_test, y_test, verbose=1 )

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))