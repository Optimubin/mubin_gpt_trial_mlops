from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

def save_model(model, filename='model.joblib'):
    joblib.dump(model, filename)

if __name__ == "__main__":
    model = train_model()
    save_model(model)
    print("model saved successfully")
