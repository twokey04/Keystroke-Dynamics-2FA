import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


class LSTMClassifier:
    def __init__(self, dataset_path: str, typing_sample: str):
        """
        Initializes the LSTMClassifier with dataset path and a typing sample.

        Args:
            dataset_path (str): The path to the dataset CSV file.
            typing_sample (list): The typing sample to be classified.
        """
        self.dataset_path = dataset_path
        self.typing_sample = typing_sample
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.X_test_scaled = None
        self.X_train_scaled = None
        self.model = None

    def load_data(self) -> None:
        """
        Loads the data from the CSV file specified by the dataset path.
        """
        self.data = pd.read_csv(self.dataset_path)
        self.X = self.data.drop(columns=['CLASS'])
        self.y = self.data['CLASS'].astype(int)

    def preprocess_data(self) -> None:
        """
        Preprocesses the data by splitting it into training and testing sets,
        scaling the features, and reshaping them for the LSTM model.
        """
        self.split_data()
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train.values)
        self.X_test_scaled = self.scaler.transform(self.X_test.values)

        self.X_train_scaled = np.reshape(self.X_train_scaled,
                                         (self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1]))
        self.X_test_scaled = np.reshape(self.X_test_scaled,
                                        (self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1]))

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def build_lstm_model(self) -> None:
        """
        Builds the LSTM model for classification.
        """
        num_classes = len(np.unique(self.y_train))
        self.model = Sequential([
            LSTM(20, input_shape=(self.X_train_scaled.shape[1], self.X_train_scaled.shape[2])),
            Dense(num_classes + 1, activation='softmax')
        ])
        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train_model(self, epochs: int = 5, batch_size: int = 16) -> None:
        """
        Trains the LSTM model on the training data.

        Args:
            epochs (int): The number of epochs to train the model.
            batch_size (int): The number of samples per gradient update.
        """
        self.model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate_model(self) -> tuple:
        """
        Evaluates the model on the training and testing sets and predicts the user ID from the typing sample.

        Returns:
            tuple: A tuple containing the predicted user ID and the test accuracy.
        """
        train_loss, train_accuracy = self.model.evaluate(self.X_train_scaled, self.y_train, verbose=0)
        test_loss, test_accuracy = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        train_accuracy *= 100
        test_accuracy *= 100

        typing_sample_scaled = self.scaler.transform([self.typing_sample])
        typing_sample_reshaped = np.reshape(typing_sample_scaled, (1, 1, self.X_train_scaled.shape[2]))
        predicted_probabilities = self.model.predict(typing_sample_reshaped)[0]

        predicted_user_index = np.argmax(predicted_probabilities)
        predicted_user_id = self.y.unique()[predicted_user_index]

        return predicted_user_id, int(test_accuracy * 100)
