import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import os
from tensorflow.keras.models import load_model
from copy import deepcopy
from tqdm import tqdm

MODEL_SAVE_PATH = 'saved_model.h5'
def load_data(stock_csv, interest_rate_csv):
    df = pd.read_csv(stock_csv)
    interest_rate_df = pd.read_csv(interest_rate_csv)

    df['Date'] = pd.to_datetime(df['Date'])
    interest_rate_df['DATE'] = pd.to_datetime(interest_rate_df['DATE'])

    df = df.rename(columns={'Date': 'DATE'})
    df = df.merge(interest_rate_df, on='DATE', how='left')

    df = df.fillna(method='ffill')

    df = df.set_index('DATE')
    return df

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=10):
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    while target_date <= last_date:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        interest_rate_values = df_subset['DGS10'].to_numpy()[:-1]
        x, y = np.column_stack((values[:-1], interest_rate_values)), values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        target_date += pd.Timedelta(days=1)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n - i} Close'] = X[:, i, 0]
        ret_df[f'Target-{n - i} Interest Rate'] = X[:, i, 1]

    ret_df['Target'] = Y

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

def train_model(X_train, y_train, X_val, y_val):
    model = Sequential([layers.Input((3, 2)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    return model

def plot_results(dates, y_true, y_pred, labels, title):
    for d, y_t, y_p, label in zip(dates, y_true, y_pred, labels):
        plt.plot(d, y_t)
        plt.plot(d, y_p)
        plt.legend(label)

    plt.title(title)
    plt.show()


import os
from tensorflow.keras.models import load_model

MODEL_SAVE_PATH = "lstm_stock_model.h5"



MODEL_SAVE_PATH = "lstm_stock_model.h5"

def split_data(dates, X, y):
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    train_data = (dates_train, X_train, y_train)
    val_data = (dates_val, X_val, y_val)
    test_data = (dates_test, X_test, y_test)

    return train_data, val_data, test_data


def build_model():
    model = Sequential([
        layers.Input((3, 2)),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])

    return model


def main():
    # Load stock data and interest rate data
    df = load_data('STAG.csv', 'US10Y.csv')

    # Create a windowed DataFrame
    windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=6)

    # Convert windowed DataFrame to date, X, and y arrays
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    # Split the data into train, validation, and test sets
    train_data, val_data, test_data = split_data(dates, X, y)

    # Load the saved model or train and save the model
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(MODEL_SAVE_PATH)
        print("Model loaded from", MODEL_SAVE_PATH)
    else:
        model = build_model()
        model = train_model(X_train, y_train, X_val, y_val)
        model.save(MODEL_SAVE_PATH)
        print("Model saved to", MODEL_SAVE_PATH)

    # Evaluate the model on the train, validation, and test sets
    plot_evaluated_model(model, train_data, val_data, test_data)

    # Predict future 30 days from April 30, 2023
    recursive_dates, recursive_predictions = predict_recursive(model, X, 30)

    # Plot the predictions
    plot_recursive_predictions(recursive_dates, recursive_predictions)

    # Plot all the data
    plot_all(model, train_data, val_data, test_data, recursive_dates, recursive_predictions)


if __name__ == "__main__":
    main()




