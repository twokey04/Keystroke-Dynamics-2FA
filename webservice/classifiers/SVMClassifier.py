import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self, typing_data_path: str, typing_pattern: str):
        """
        Initializes the SVMClassifier with typing data path and a typing pattern.

        Args:
            typing_data_path (str): The path to the typing data CSV file.
            typing_pattern (list): The typing pattern to be classified.
        """
        self.data = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.model = None
        self.typing_data_path = typing_data_path
        self.typing_pattern = typing_pattern
        self.load_data()
        self.split_data()
        self.train_model()

    def load_data(self) -> None:
        """
        Loads the typing data from the CSV file specified by the typing data path.
        """
        self.data = pd.read_csv(self.typing_data_path, keep_default_na=False)

    def split_data(self) -> None:
        """
        Splits the typing data into training and testing sets.
        """
        x = self.data.drop(columns=["CLASS"])  # Input features
        y = self.data["CLASS"]  # Labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def train_model(self) -> None:
        """
        Trains the SVM model on the training data.
        """
        self.model = SVC(kernel='linear')
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self) -> tuple:
        """
        Evaluates the model using cross-validation and returns the mean accuracy and standard deviation.

        Returns:
            tuple: A tuple containing the mean accuracy and standard deviation of cross-validation scores.
        """
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=4)

        accuracy = cv_scores.mean() * 100
        std_dev = cv_scores.std() * 100

        return accuracy, std_dev

    def authenticate(self) -> int:
        """
        Predicts the class label of the given typing pattern.

        Returns:
            str: The predicted class label.
        """
        typing_pattern_df = pd.DataFrame([self.typing_pattern])
        prediction = self.model.predict(typing_pattern_df)
        return int(prediction[0])
