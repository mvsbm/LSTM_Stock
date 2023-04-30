import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from pylab import rcParams


def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


def load_data():
    stock_dataset_train = pd.read_csv('STAG.csv')
    interest_rate_dataset = pd.read_csv('US10Y.csv')

    # Merge datasets on the 'Date' column using inner join
    merged_dataset = stock_dataset_train.merge(interest_rate_dataset, on='Date', how='inner')

    cols = list(merged_dataset)[1:7]

    datelist_train = list(merged_dataset['Date'])
    datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

    return merged_dataset, cols, datelist_train


def preprocess_data(stock_dataset_train, cols):
    stock_dataset_train = stock_dataset_train[cols].astype(str)
    for i in cols:
        for j in range(0, len(stock_dataset_train)):
            stock_dataset_train[i][j] = stock_dataset_train[i][j].replace(',', '')

    stock_dataset_train = stock_dataset_train.astype(float)

    return stock_dataset_train


def create_training_set(stock_dataset_train):
    # Using multiple features (predictors)
    training_set = stock_dataset_train.values

    # Feature Scaling
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)

    sc_predict = StandardScaler()
    sc_predict.fit_transform(training_set[:, 0:1])

    return training_set, training_set_scaled, sc, sc_predict


def prepare_data_structure(training_set_scaled, stock_dataset_train):
    X_train = []
    y_train = []

    n_future = 60
    n_past = 90

    for i in range(n_past, len(training_set_scaled) - n_future + 1):
        X_train.append(training_set_scaled[i - n_past:i, 0:stock_dataset_train.shape[1] - 1])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train, n_future, n_past, len(training_set_scaled) - n_future - n_past + 1


def create_model(stock_dataset_train, n_past):
    model = Sequential()

    model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, stock_dataset_train.shape[1] - 1)))
    model.add(LSTM(units=10, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

    return model


def train_model(model, X_train, y_train):
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tb = TensorBoard('logs')

    history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)

    return history


def predict_stock_prices(model, X_train, y_train, n_future, n_past, sc_predict, datelist_train):
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()
    datelist_future_ = [this_timestamp.date() for this_timestamp in datelist_future]

    predictions_future = model.predict(X_train[-n_future:])
    predictions_train = model.predict(X_train[n_past:])

    y_pred_future = sc_predict.inverse_transform(predictions_future)
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    return y_pred_future, y_pred_train, datelist_future


def plot_predictions(y_pred_future, y_pred_train, datelist_train, datelist_future, stock_dataset_train, n_past,
                     n_future, valid_range):
    plt.figure(figsize=(15, 5))

    # Reshape y_pred_train and plot it
    y_pred_train_reshaped = y_pred_train.reshape(-1)
    plt.plot(datelist_train[-valid_range:], y_pred_train_reshaped, color='orange', label='Predicted Stock Price')

    # Create a DataFrame to store the predicted stock prices
    future_df = pd.DataFrame(y_pred_future, index=datelist_future, columns=['Prediction'])

    # Calculate the start date for plotting: 3 years before the last date in the training set
    START_DATE_FOR_PLOTTING = stock_dataset_train.index[-1] - dt.timedelta(days=3 * 365)

    # Plot actual stock prices
    plt.plot(stock_dataset_train.loc[START_DATE_FOR_PLOTTING:].index,
             stock_dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')

    # Plot predicted stock prices for the next month
    plt.plot(future_df.index, future_df['Prediction'], color='r', label='Predicted Stock Price for Next Month')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Predictions')
    plt.show()

    # Save predictions to a CSV file
    future_df.to_csv('predictions.csv')


def main():
    stock_dataset_train, cols, datelist_train = load_data()
    stock_dataset_train = preprocess_data(stock_dataset_train, cols)
    training_set, training_set_scaled, sc, sc_predict = create_training_set(stock_dataset_train)
    X_train, y_train, n_future, n_past, valid_range = prepare_data_structure(training_set_scaled, stock_dataset_train)
    model = create_model(stock_dataset_train, n_past)
    history = train_model(model, X_train, y_train)
    y_pred_future, y_pred_train, datelist_future = predict_stock_prices(model, X_train, y_train, n_future, n_past, sc_predict, datelist_train)
    plot_predictions(y_pred_future, y_pred_train, datelist_train, datelist_future, stock_dataset_train, n_past, n_future, valid_range)

if __name__ == '__main__':
    main()


