import joblib
import numpy as np
import sys
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier
from logger import logging
from exception import CustomException
from utils import evaluate_models
class ModelConfig:
    def __init__(self):
            self.trained_model_file_path = "voice_disease_model.joblib"

class ModelTrainer:
    def __init__(self):
            self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self, train_array, test_array):
            try:
                logging.info("🔀 Splitting training and test arrays")
                X_train, y_train, X_test, y_test = (
                    train_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, :-1],
                    test_array[:, -1]
                )

                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    #"CatBoost": CatBoostClassifier(verbose=False),
                    "AdaBoost": AdaBoostClassifier(),
                }

                params = {
                    "Random Forest": {"n_estimators": [64, 128]},
                    "Decision Tree": {"criterion": ["gini", "entropy"]},
                    "Gradient Boosting": {"n_estimators": [64, 128], "learning_rate": [0.01, 0.1]},
                    "Logistic Regression": {"C": [0.1, 1.0, 10]},
                    "CatBoost": {"depth": [6, 8], "learning_rate": [0.01, 0.1], "iterations": [50]},
                    "AdaBoost": {"n_estimators": [64, 128], "learning_rate": [0.01, 0.1]}
                }

                model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

                best_model_score = max(model_report.values())
                best_model_name = max(model_report, key=model_report.get)
                best_model = models[best_model_name]

                logging.info(f"✅ Best model: {best_model_name} with accuracy {best_model_score:.4f}")

                if best_model_score < 0.6:
                    raise Exception("❌ No good model found. Accuracy below threshold.")

                joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
                logging.info("✅ Model saved successfully.")

                # Optional: print evaluation
                y_pred = best_model.fit(X_train, y_train).predict(X_test)
                print("\n📊 Final Evaluation:")
                print("Accuracy:", accuracy_score(y_test, y_pred))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                print("Classification Report:\n", classification_report(y_test, y_pred))

                return best_model

            except Exception as e:
                raise CustomException(str(e),sys)
