from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import micromlgen

def train(X_train, y_train, C=1000, gamma='scale', kernel='rbf'):
    """
    Entrena un modelo Support Vector Machine (SVM).
    """
    model = SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    """
    Evalúa el modelo y devuelve la exactitud.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

def export_to_c(model, filename="svm_model.h"):
    """
    Exporta el modelo SVM a código C++ utilizando micromlgen.
    """
    c_code = micromlgen.port(model)
    with open(filename, "w") as f:
        f.write(c_code)
    return c_code
