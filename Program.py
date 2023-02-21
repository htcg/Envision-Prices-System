import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

# from tkinter import *
# Explicit imports to satisfy Flake8


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\TRIKSI'S FILES\MODSIM\Price Prediction\Price Prediction\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def name_google():
    # Get the stock data
    stock_data = yf.download('GOOGL', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse

    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()


def name_meta():
    # Get the stock data
    stock_data = yf.download('META', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse

    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()


def name_microsoft():
    # Get the stock data
    stock_data = yf.download('MSFT', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse

    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()


def name_tesla():
    # Get the stock data
    stock_data = yf.download('TSLA', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse

    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()


def name_apple():
    # Get the stock data
    stock_data = yf.download('AAPL', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse

    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()


def name_amazon():
    # Get the stock data
    stock_data = yf.download('AMZN', start=dt.datetime(2021, 1, 1), end=dt.datetime.now())
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History', fontsize=13)
    plt.plot(stock_data['Close'])
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Prices ($)', fontsize=13)

    # Create a new dataframe with only the 'Close column'
    close_prices = stock_data['Close']
    # Convert the dataframe to a numpy array
    values = close_prices.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(values) * 0.8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    # Create the training data set
    train_data = scaled_data[0: training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Get the models prediction  price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    rmse


    # Plot the data
    data = stock_data.filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Amazon Prediction Model', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Close Price USD ($)', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.grid(axis='y')
    plt.show()



window = Tk()
window.title("Envision Prices")
photo = PhotoImage(file="D:\TRIKSI'S FILES\MODSIM\Price Prediction\Price Prediction\\build\\assets\\icon.png")
window.iconphoto(False, photo)

window.geometry("1300x800")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=800,
    width=1300,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    325.0,
    45.0,
    fill="#3D5A80",
    outline="")

canvas.create_rectangle(
    325.0,
    0.0,
    650.0,
    45.0,
    fill="#98C1D9",
    outline="")

canvas.create_rectangle(
    650.0,
    0.0,
    975.0,
    45.0,
    fill="#8DE1E4",
    outline="")

canvas.create_rectangle(
    975.0,
    0.0,
    1300.0,
    45.0,
    fill="#293241",
    outline="")

canvas.create_rectangle(
    0.0,
    755.0,
    325.0,
    800.0,
    fill="#293241",
    outline="")

canvas.create_rectangle(
    325.0,
    755.0,
    650.0,
    800.0,
    fill="#8CE1E4",
    outline="")

canvas.create_rectangle(
    650.0,
    755.0,
    975.0,
    800.0,
    fill="#98C1D9",
    outline="")

canvas.create_rectangle(
    975.0,
    755.0,
    1300.0,
    800.0,
    fill="#3D5A80",
    outline="")

canvas.create_text(
    326.0,
    129.0,
    anchor="nw",
    text="In what company are you interested?",
    fill="#000000",
    font=("Inter Bold", 36 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    963.0,
    367.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=name_google,
    relief="flat"
)
button_1.place(
    x=916.0,
    y=451.0,
    width=95.0,
    height=19.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    830.0,
    559.0,
    image=image_image_2
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=name_meta,
    relief="flat"
)
button_2.place(
    x=796.0,
    y=641.0,
    width=72.0,
    height=19.0
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    703.0,
    367.0,
    image=image_image_3
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=name_microsoft,
    relief="flat"
)
button_3.place(
    x=651.0,
    y=451.0,
    width=106.0,
    height=19.0
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    566.0,
    559.0,
    image=image_image_4
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=name_tesla,
    relief="flat"
)
button_4.place(
    x=530.0,
    y=641.0,
    width=74.0,
    height=19.0
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    443.0,
    367.0,
    image=image_image_5
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=name_apple,
    relief="flat"
)
button_5.place(
    x=403.0,
    y=451.0,
    width=79.0,
    height=19.0
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))

image_6 = canvas.create_image(
    306.0,
    559.0,
    image=image_image_6
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))

button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=name_amazon,
    relief="flat"
)

button_6.place(
    x=261.0,
    y=641.0,
    width=97.0,
    height=19.0
)

window.resizable(False, False)
window.mainloop()
