import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify

from database.database_utils import *
from classifiers.SVMClassifier import SVMClassifier
from classifiers.KNNClassifier import KNNClassifier
from classifiers.LSTMClassifier import LSTMClassifier
from classifiers.HMMClassifier import HMMClassifier

import pandas as pd
import datetime
import csv

TYPING_DATA_PATH = 'biometrics/biometrics.csv'
LOGS_PATH = 'results.log'

app = Flask(__name__, static_folder='./static')


@app.route('/')
def home():
    return render_template('./home/home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('./register/register.html')
    elif request.method == 'POST':
        response = dict(request.get_json())
        username = response['username']
        password = response['password']
        user_id, result = add_user_and_passw(username, password)

        if result:
            return jsonify({'register_code': 'UserRegistrySuccess', 'id_user': user_id})
        else:
            return jsonify({'register_code': 'UsernameAlreadyExist'})


@app.route('/register/biometrics', methods=['POST'])
def biometrics():
    if request.method == 'POST':
        response = dict(request.get_json())
        user_id = response['user_id']
        data = response['data']

        if not isinstance(data, list):
            data = [data]

        if len(data) != 85:
            return jsonify({"needRefresh": True})

        data.append(user_id)

        result = append_to_csv(data)

        return jsonify({'biometrics_code': result})


@app.route('/train/biometrics', methods=['POST'])
def train():
    if request.method == 'POST':
        response = dict(request.get_json())
        username = response['username']
        data = response['data']

        if len(data) != 85:
            return jsonify({"needRefresh": True})

        user_id = get_user_id(username)
        if user_id is not None:
            if not isinstance(data, list):
                data = [data]

            data.append(user_id)
            result = append_to_csv(data)

            return jsonify({'biometrics_code': result})


@app.route('/login', methods=['GET'])
def login():
    return render_template('./login/login.html')


@app.route('/login/auth1', methods=['POST'])
def auth1():
    try:
        response = dict(request.get_json())
        username = response['username']
        password = response['password']

        user_id, result, user_id = check_user_and_passw(username, password)

        if result:
            return jsonify({'auth1_code': 'success', 'id_user': user_id})
        else:
            if user_id == 3:
                return jsonify({'auth1_code': 'UsernameNotExist'})
            elif user_id == 1:
                return jsonify({'auth1_code': 'PasswordIsWrong'})
    except Exception as e:
        print(e)
        return jsonify({'auth1_code': 'ServerError'})


@app.route('/login/auth2', methods=['POST'])
def auth2():
    response = dict(request.get_json())
    typing_sample = response['typing_data']
    user_id = response['user_id']

    if len(typing_sample) != 85:
        return jsonify({"needRefresh": True})

    def svm_task():
        svm_detector = SVMClassifier(TYPING_DATA_PATH, typing_sample)
        svm_accuracy, svm_std = svm_detector.evaluate_model()
        svm_prediction = svm_detector.authenticate()
        return 'SVM', user_id, svm_prediction, svm_accuracy

    def knn_task():
        knn_classifier = KNNClassifier(TYPING_DATA_PATH, typing_sample)
        knn_classifier.load_data()
        knn_classifier.preprocess_data()
        knn_classifier.split_data()
        knn_classifier.train_knn_models()
        knn_classifier.hyperparameter_tuning()
        (knn_manhattan_prediction, knn_manhattan_mean_accuracy, knn_euclidian_prediction,
         knn_euclidian_mean_accuracy,) = knn_classifier.evaluate_models()
        return ('KNN', user_id, knn_manhattan_prediction, knn_manhattan_mean_accuracy, knn_euclidian_prediction,
                knn_euclidian_mean_accuracy)

    def lstm_task():
        lstm_classifier = LSTMClassifier(TYPING_DATA_PATH, typing_sample)
        lstm_classifier.load_data()
        lstm_classifier.preprocess_data()
        lstm_classifier.build_lstm_model()
        lstm_classifier.train_model()
        lstm_prediction, lstm_accuracy = lstm_classifier.evaluate_model()
        return 'LSTM', user_id, lstm_prediction, lstm_accuracy

    def hmm_task():
        hmm_classifier = HMMClassifier(TYPING_DATA_PATH, typing_sample)
        hmm_classifier.load_data()
        hmm_classifier.preprocess_data()
        hmm_classifier.build_hmm_model(n_components=3)
        hmm_classifier.train_model()
        hmm_pred_series = hmm_classifier.evaluate_model()
        hmm_prediction = hmm_pred_series.iloc[0] if isinstance(hmm_pred_series, pd.Series) else hmm_pred_series
        return 'HMM', user_id, hmm_prediction

    def log_results(real_user, pred_user, algorithm, match):
        date_of_execution = datetime.datetime.now()
        formatted_date_of_execution = date_of_execution.strftime("%d/%m/%Y %H:%M:%S")
        with open(LOGS_PATH, 'a') as algorithm_logs:
            algorithm_logs.write('[RESULT] Real user: ')
            algorithm_logs.write(str(real_user))
            algorithm_logs.write(' | Predicted user: ')
            algorithm_logs.write(str(pred_user))
            algorithm_logs.write(' | Algorithm: ')
            algorithm_logs.write(algorithm)
            algorithm_logs.write(' | Match: ')
            algorithm_logs.write(str(match))
            algorithm_logs.write(' | Date of execution: ')
            algorithm_logs.write(formatted_date_of_execution)
            algorithm_logs.write('\n')

    with ThreadPoolExecutor() as executor:
        svm_result = executor.submit(svm_task)
        knn_result = executor.submit(knn_task)
        lstm_result = executor.submit(lstm_task)
        hmm_result = executor.submit(hmm_task)

        svm_alg, svm_real, svm_pred, svm_acc = svm_result.result()
        knn_alg, knn_real, knn_manhattan_pred, knn_manhattan_mean_acc, knn_euclidian_pred, knn_euclidian_mean_acc = \
            knn_result.result()
        lstm_alg, lstm_real, lstm_pred, lstm_acc = lstm_result.result()
        hmm_alg, hmm_real, hmm_pred = hmm_result.result()

    log_results(svm_real, svm_pred, svm_alg, svm_real == svm_pred)
    log_results(knn_real, knn_manhattan_pred, 'KNN Manhattan', knn_real == knn_manhattan_pred)
    log_results(knn_real, knn_euclidian_pred, 'KNN Euclidean', knn_real == knn_euclidian_pred)
    log_results(lstm_real, lstm_pred, lstm_alg, lstm_real == lstm_pred)
    log_results(hmm_real, hmm_pred, hmm_alg, hmm_real == hmm_pred)

    return jsonify({
        'user_id': str(get_user_and_passw(int(user_id))[0]),
        'svm': {'prediction': str(get_user_and_passw(svm_pred)[0]),
                'match': bool(svm_real == svm_pred)},
        'knn_manhattan': {'prediction': str(get_user_and_passw(knn_manhattan_pred)[0]),
                          'match': bool(knn_real == knn_manhattan_pred)},
        'knn_euclidean': {'prediction': str(get_user_and_passw(knn_euclidian_pred)[0]),
                          'match': bool(knn_real == knn_euclidian_pred)},
        'lstm': {'prediction': str(get_user_and_passw(lstm_pred)[0]),
                 'match': bool(lstm_real == lstm_pred)},
        'hmm': {'prediction': str(get_user_and_passw(hmm_pred)[0]),
                'match': bool(hmm_pred == user_id)}
    })


@app.route('/train', methods=['GET', 'POST'])
def train_biometrics():
    if request.method == 'GET':
        return render_template('./train/train.html')


def append_to_csv(data):
    try:
        with open(TYPING_DATA_PATH, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        return 'Success'
    except Exception as e:
        print(e)
        return 'Biometrics could not be registered'


def test_file_access(file_path):
    try:
        with open(file_path, 'r') as _:
            print(f'Success: {file_path} can be opened.')
    except FileNotFoundError:
        print(f'Error: {file_path} does not exist.')
        exit()
    except IOError:
        print(f'Error: {file_path} cannot be accessed.')
        exit()


if __name__ == '__main__':
    current_working_directory = os.getcwd()
    print(f'[INFO] Current working directory: {current_working_directory}')
    if current_working_directory == '/home/twokey/PycharmProjects/KsDynAUTH':
        os.chdir('/home/twokey/PycharmProjects/KsDynAUTH/webservice')
        print(f'[INFO] Changed current working directory: {os.getcwd()}')

    test_connection_db()
    test_file_access(TYPING_DATA_PATH)
    test_file_access(LOGS_PATH)

    datetime_now = datetime.datetime.now()
    formatted_datetime_now = datetime_now.strftime("%d/%m/%Y %H:%M:%S")

    with open(LOGS_PATH, 'a') as logs:
        logs.write(f'\n[INFO] Server online at {formatted_datetime_now}\n')

    app.run(host='127.0.0.1', debug=True, port=3000)
