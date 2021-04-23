# Weather Prediction Project
# Authors: Mohammad Alhusseini and Runxiong Dong
# The authors contributed equally to this code.

# Package Imports
import numpy as np
from time import time
import datetime as dt

# Plotting Tools Imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.use('Agg')

# Keras Imports
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.callbacks import TensorBoard

# Dictionaries to store Normalization constants
# these must be used to denormalize the predicted values
means = {}
maxxs = {}
minxs = {}

# Function to normalize the data
def normalize(name, X):
    tmp = np.copy(X)
    for col in range(tmp.shape[1]):
        mean = np.mean(tmp[:, col]) + 0.00001
        maxx = np.max(tmp[:, col])
        minx = np.min(tmp[:, col])
        tmp[:, col] = tmp[:, col] - mean
        tmp[:, col] = tmp[:, col] / (maxx - minx)
        means[name + str(col)] = mean
        maxxs[name + str(col)] = maxx
        minxs[name + str(col)] = minx
    return tmp


# Function to de-normalize the predicted value
def denormalize(name, X):
    tmp = np.copy(X)
    for col in range(tmp.shape[1]):
        mean = means[name + str(col)]
        maxx = maxxs[name + str(col)]
        minx = minxs[name + str(col)]
        tmp[:, col] = tmp[:, col] * (maxx - minx)
        tmp[:, col] = tmp[:, col] + mean
    return tmp


# Function to generate a seq2seq model
# the function takes the following arguments
# n_input: number of input features
# n_output: number of output features
# n_units: number of units in our RNN Cells
# returns model used for training, model for encoder,
# model for decoder
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(n_input, 1))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(n_output, 1))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units, ))
    decoder_state_input_c = Input(shape=(n_units, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

# function that predicts the next n_steps values
# using a source sequence. The parameters
# infec - model for encoder
# infdec - model for decoder
# are the models returned by define_models function
def predict_sequence(infenc, infdec, source, n_steps):
    state = infenc.predict(source)
    target_seq = np.zeros((1, 12, 1))
    output = list()
    for t in range(n_steps):
        yhat, h, c = infdec.predict([target_seq] + state)
        output.append(yhat)
        state = [h, c]
        target_seq = yhat
    return np.array(output)

if __name__ == "__main__":

    predict_days = 7  # number of future days to predict
    use_past = 21  # number of past days to use for prediction

    # load the data from the csv files, select the columns that will be
    # used for our sequence, and skip the first row (column titles)
    LosAngeles = np.loadtxt(
        'data/LosAngelesData.csv', delimiter=',', skiprows=1, usecols=(1, 3, 4, 5))
    LasVegas = np.loadtxt(
        'data/LasVegasData.csv', delimiter=',', skiprows=1, usecols=(1, 3, 4, 5))
    Phoenix = np.loadtxt(
        'data/PhoenixData.csv', delimiter=',', skiprows=1, usecols=(1, 3, 4, 5))

    # Normalize each feature for each city indepently
    # and expand with a new axis
    LosAngeles = np.expand_dims(LosAngeles, axis=2)
    LosAngelesN = normalize('la', LosAngeles)

    LasVegas = np.expand_dims(LasVegas, axis=2)
    LasVegasN = normalize('lv', LasVegas)

    Phoenix = np.expand_dims(Phoenix, axis=2)
    PhoenixN = normalize('ph', Phoenix)

    # Combine the data into a single array
    # with 12 features (4 per city)
    CombinedData = np.concatenate((LosAngelesN, LasVegasN, PhoenixN), axis=1)

    # split data 80/20 for training/validation and testing
    m = int(len(CombinedData) * .8)
    train = CombinedData[:m]
    test = CombinedData[m:]

    # define our training, encoder and decoder models
    trainModel, endModel, decModel = define_models_gru(12, 12, 100)

    # compile the training model
    trainModel.compile(loss="mean_squared_error", optimizer="adam")

    # encoder inputs and decoder outputs have same shape,
    # but are offset by the number of prediction days
    enc_inp = train[:-predict_days]
    dec_out = train[predict_days:]
    # decoder input also has the same shape, but is
    # offset from encoder input by 1 time step
    dec_inp = np.concatenate((np.zeros(12).reshape(1, 12, 1), enc_inp[:-1]), axis=0)

    # define a tensorboard instance and log directory
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # train out model
    trainModel.fit(
        [enc_inp, dec_inp],
        dec_out,
        batch_size=32,
        epochs=1,
        validation_split=0.2,
        callbacks=[tensorboard])

    # predict the last predict_days of our test data using the previous use_past days
    prediction = predict_sequence(
        endModel, decModel, test[-(use_past + predict_days):-predict_days], predict_days)
    prediction = np.squeeze(prediction, axis=1)

    # split predicitions for each city, so we can denormalize using
    # the dictionary values for each city

    cities = np.split(prediction, 3, axis=1)
    LAPred, LVPred, PHPred = cities

    # denormalize each city's predicted values
    LAPred = denormalize('la', LAPred)
    LVPred = denormalize('lv', LVPred)
    PHPred = denormalize('ph', PHPred)

    # define our expected and past values
    LAExp = LosAngeles[-predict_days:]
    LVExp = LasVegas[-predict_days:]
    PHExp = Phoenix[-predict_days:]

    LApast = LosAngeles[-(use_past + predict_days):-predict_days]
    LVpast = LasVegas[-(use_past + predict_days):-predict_days]
    PHpast = Phoenix[-(use_past + predict_days):-predict_days]

    # to use in plotting, we grab the dates column
    # and convert to python datetime object
    date_column = list(
        np.loadtxt(
            'data/LosAngelesData.csv', delimiter=',', skiprows=1, usecols=0, dtype=str))
    dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in date_column]

    # List of values for each city and name of city
    city_values = [[LApast, LAExp, LAPred, 'Los Angeles'],
                   [LVpast, LVExp, LVPred, "Las Vegas"], [PHpast, PHExp, PHPred, 'Phoenix']]

    # dictionary of feature names
    feature_dictionary = {0: "AWND", 1: "TMAX", 2: "TMIN", 3: "WDF2"}

    # making the plots for features of each city
    for city in city_values:
        for i in range(0, 4):
            fig = plt.figure(figsize=(12, 3))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.plot(
                dates[-(use_past + predict_days):-predict_days],
                city[0][:, i, :],
                "o--b",
                label="seen")

            # plt.plot(range(use_past,use_past+predict_days),PHExp[:,i,:],"x--b",label="expec")
            plt.plot(dates[-predict_days:], city[1][:, i, :], "x--b", label="expec")

            # plt.plot(range(use_past,predict_days+use_past),PHPred[:,i,:],"o--y",label="pred")
            plt.plot(dates[-predict_days:], city[2][:, i, :], "o--y", label="pred")
            # plt.gcf().autofmt_xdate()
            plt.xticks(fontsize=7, rotation=85)
            plt.legend(loc='best')
            plt.title("Predictions vs True Values for {} {}".format(
                city[3], feature_dictionary[i]))
            fig.savefig(
                'plots/predict_{}_{}.png'.format(city[3].replace(" ", ""),
                                                 feature_dictionary[i]),
                bbox_inches='tight',dpi=300)
            plt.close()
