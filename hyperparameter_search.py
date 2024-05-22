import warnings
import os
import sys
from time import time
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


class FaceClassifier:
    def __init__(self):
        self.faces = fetch_lfw_people(data_home='.', min_faces_per_person=70, resize=0.4)
        self.X, self.y = self.faces.data, self.faces.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25,
                                                                                random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # TODO Q2.10 - Task 1 - Initialize the MLPClassifier with the some hyperparameters
        # -------------------------------
        # FIXME
        self.mlp = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', max_iter=200, learning_rate_init=0.001,
                                 random_state=42)
        # -------------------------------

    def grid_search(self):
        # TODO Q2.10 - Task 2 - Define the hyperparameter grid search for GridSearchCV
        # Replace the NotImplemented error with the hyperparameter grid search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
            'learning_rate_init': [0.01, 0.001, 0.0001],
        }
        grid_search = GridSearchCV(estimator=self.mlp, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)

        tic = time()
        grid_search.fit(self.X_train, self.y_train)

        return grid_search, time() - tic
        # raise NotImplementedError

    def random_search(self):
        # TODO Q2.10 - Task 3 - Define the hyperparameter random search for RandomizedSearchCV
        # Replace the NotImplemented error with the hyperparameter random search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
            'learning_rate_init': [0.01, 0.001, 0.0001],
        }
        random_search = RandomizedSearchCV(estimator=self.mlp, param_distributions=param_grid, n_iter=10, n_jobs=-1,
                                           cv=5, verbose=2, random_state=42)

        tic = time()
        random_search.fit(self.X_train, self.y_train)

        return random_search, time() - tic

    def bayes_search(self):
        # TODO Q2.10 - Task 4 - Define the hyperparameter bayes search for BayesSearchCV
        # Replace the NotImplemented error with the hyperparameter bayes search
        search_spaces = {
            'hidden_layer_sizes': Integer(50, 200),
            'learning_rate_init': Real(0.0001, 0.01, prior='log-uniform'),
        }
        bayes_search = BayesSearchCV(estimator=self.mlp, search_spaces=search_spaces, n_iter=32, n_jobs=-1, cv=5,
                                     verbose=2,
                                     random_state=42)

        tic = time()
        bayes_search.fit(self.X_train, self.y_train)

        return bayes_search, time() - tic

    def evaluate(self, model):
        # TODO Q2.10 - Task 5 - Evaluate the model and print the classification report and plot the confusion matrix
        # Replace the NotImplemented error with the evaluation code
        y_pred = model.best_estimator_.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # confu_mat = confusion_matrix(self.y_test, y_pred)

        num_class = max(self.y_test) - min(self.y_test) + 1
        confu_mat = np.array([[0] * num_class for _ in range(num_class)])

        for true, pred in zip(self.y_test, y_pred):
            confu_mat[true, pred] += 1

        plt.figure(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=confu_mat)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
        # raise NotImplementedError


def perform_hyperparameter_tuning():
    classifier = FaceClassifier()
    grid_search, grid_search_time = classifier.grid_search()
    random_search, random_search_time = classifier.random_search()
    bayes_search, bayes_search_time = classifier.bayes_search()

    print('Grid Search Time:', grid_search_time)
    print('Grid Search Best Params:', grid_search.best_params_)
    grid_search_test_score = grid_search.score(classifier.X_test, classifier.y_test)
    print('Grid Search Test Score:', grid_search_test_score)
    classifier.evaluate(grid_search)

    print('Random Search Time:', random_search_time)
    print('Random Search Best Params:', random_search.best_params_)
    random_search_test_score = random_search.score(classifier.X_test, classifier.y_test)
    print('Random Search Test Score:', random_search_test_score)
    classifier.evaluate(random_search)

    print('Bayes Search Time:', bayes_search_time)
    print('Bayes Search Best Params:', bayes_search.best_params_)
    bayes_search_test_score = bayes_search.score(classifier.X_test, classifier.y_test)
    print('Bayes Search Test Score:', bayes_search_test_score)
    classifier.evaluate(bayes_search)

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.bar(['Grid Search', 'Random Search', 'Bayes Search'], [grid_search_time, random_search_time, bayes_search_time],
            log=True)
    plt.ylabel('Time (log scale)')
    plt.title('Comparison of Running Times')

    plt.subplot(122)
    plt.bar(['Grid Search', 'Random Search', 'Bayes Search'],
            [grid_search_test_score, random_search_test_score, bayes_search_test_score])
    plt.ylabel('Best Score')
    plt.title('Comparison of Best Test Scores')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    perform_hyperparameter_tuning()
