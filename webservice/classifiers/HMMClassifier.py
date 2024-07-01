import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


class HMMClassifier:
    def __init__(self, dataset_path: str, typing_sample: str):
        """
        Initializes the HMMClassifier with dataset path and a typing sample.

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
        and scaling the features.
        """
        self.split_data()
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train.values)

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

    def build_hmm_model(self, n_components: int = 3) -> None:
        """
        Builds the HMM model for classification.

        Args:
            n_components (int): The number of states in the HMM.
        """
        self.model = hmm.GaussianHMM(n_components=n_components)

    def train_model(self) -> None:
        """
        Trains the HMM model on the training data.
        """
        self.model.fit(self.X_train_scaled)

    def evaluate_model(self) -> int:
        """
        Evaluates the model and predicts the user ID from the typing sample.

        Returns:
            int: The predicted user ID.
        """
        typing_sample_scaled = self.scaler.transform([self.typing_sample])
        predicted_user_id = self.model.predict(typing_sample_scaled)
        return self.y[predicted_user_id[0]]
