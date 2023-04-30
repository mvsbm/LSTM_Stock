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

    stock_dataset_train['DGS10'] = interest_rate_dataset['DGS10']
    cols = list(stock_dataset_train)[1:7]

    datelist_train = list(stock_dataset_train['Date'])
    datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

    return stock_dataset_train, cols, datelist_train

def preprocess_data(stock_dataset_train, cols):
    stock_dataset_train = stock_dataset_train[cols].astype(str)
    for i in cols:
        for j in range(0, len(stock_dataset_train)):
            stock_dataset_train[i][j] = stock_dataset_train[i][j].replace(',', '')

    stock_dataset_train = stock_dataset_train.astype(float)

    return stock_dataset_train

def create_training_set(stock_dataset_train):
    training_set = stock_dataset_train.as_matrix()

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

    return X_train, y_train, n_future, n_past

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

def predict_stock_prices(model, X_train, y_train, n_future, n_past, sc_predict):
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()
    datelist_future_ = [this_timestamp.date() for this_timestamp in datelist_future]

    predictions_future = model.predict(X_train[-n_future:])
    predictions_train = model.predict(X_train[n_past:])

    y_pred_future = sc_predict.inverse_transform(predictions_future)
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    return y_pred_future, y_pred_train, datelist_future

def plot_predictions(y_pred_future, y_pred_train, datelist_train, datelist_future, dataset_train):
    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

    PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

    rcParams['figure.figsize'] = 14, 5

    START_DATE_FOR_PLOTTING = '2012-06-01'

    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted Stock Price')
    plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
    plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')

    plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predictions and Actual Stock Prices', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.show()

def main():
    stock_dataset_train, cols, datelist_train = load_data()
    stock_dataset_train = preprocess_data(stock_dataset_train, cols)
    training_set, training_set_scaled, sc, sc_predict = create_training_set(stock_dataset_train)
    X_train, y_train, n_future, n_past = prepare_data_structure(training_set_scaled, stock_dataset_train)
    model = create_model(stock_dataset_train, n_past)
    history = train_model(model, X_train, y_train)
    y_pred_future, y_pred_train, datelist_future = predict_stock_prices(model, X_train, y_train, n_future, n_past, sc_predict)
    plot_predictions(y_pred_future, y_pred_train, datelist_train, datelist_future, stock_dataset_train)

if __name__ == '__main__':
    main()

