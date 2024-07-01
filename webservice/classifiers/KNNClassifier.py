import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, dataset_path: str, typing_sample: str):
        """
        Initializes the KNNClassifier with dataset path and a typing sample.

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
        self.knn_manhattan = None
        self.knn_euclidean = None

    def load_data(self) -> None:
        """
        Loads the data from the CSV file specified by the dataset path.
        """
        self.data = pd.read_csv(self.dataset_path)
        self.X = self.data.drop(columns=['CLASS'])
        self.y = self.data['CLASS']

    def preprocess_data(self) -> None:
        """
        Preprocesses the data by scaling the features.
        """
        self.X = self.scaler.fit_transform(self.X)

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

    def train_knn_models(self) -> None:
        """
        Trains KNN models with Manhattan and Euclidean distances.
        """
        self.knn_manhattan = KNeighborsClassifier(metric='manhattan')
        self.knn_manhattan.fit(self.X_train, self.y_train)

        self.knn_euclidean = KNeighborsClassifier(metric='euclidean')
        self.knn_euclidean.fit(self.X_train, self.y_train)

    def hyperparameter_tuning(self) -> None:
        """
        Performs hyperparameter tuning for KNN models using GridSearchCV.
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }

        grid_manhattan = GridSearchCV(self.knn_manhattan, param_grid, cv=4)
        grid_manhattan.fit(self.X_train, self.y_train)

        grid_euclidean = GridSearchCV(self.knn_euclidean, param_grid, cv=4)
        grid_euclidean.fit(self.X_train, self.y_train)

        self.knn_manhattan = grid_manhattan.best_estimator_
        self.knn_euclidean = grid_euclidean.best_estimator_

    def evaluate_models(self) -> tuple:
        """
        Evaluates the KNN models using cross-validation and predicts class labels for the typing sample.

        Returns:
            tuple: A tuple containing predictions and mean accuracies for Manhattan and Euclidean models.
        """
        manhattan_cv_scores = cross_val_score(self.knn_manhattan, self.X, self.y, cv=4)
        euclidean_cv_scores = cross_val_score(self.knn_euclidean, self.X, self.y, cv=4)

        manhattan_mean_accuracy = manhattan_cv_scores.mean() * 100
        euclidean_mean_accuracy = euclidean_cv_scores.mean() * 100

        typing_sample_scaled = self.scaler.transform([self.typing_sample])
        manhattan_prediction = self.knn_manhattan.predict(typing_sample_scaled)
        euclidean_prediction = self.knn_euclidean.predict(typing_sample_scaled)

        return (manhattan_prediction[0], int(manhattan_mean_accuracy), euclidean_prediction[0],
                int(euclidean_mean_accuracy))
