
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def build_model(input_shape):
    """
    Builds and returns the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    # Load the processed data
    data = pd.read_csv('data/processed/processed_data.csv')

    # For simplicity, we'll use only the 'Close' price for prediction
    closed_price = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(closed_price)

    # Split the data into training and testing sets
    training_size = int(len(scaled_price) * 0.8)
    test_size = len(scaled_price) - training_size
    train_data, test_data = scaled_price[0:training_size, :], scaled_price[training_size:len(scaled_price), :1]

    # Create the training and testing datasets
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the model
    model = build_model((X_train.shape[1], 1))

    # Define a checkpoint to save the best model
    checkpoint = ModelCheckpoint('model/lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, callbacks=[checkpoint])
