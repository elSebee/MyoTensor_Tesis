from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import micromlgen

def train(X_train, y_train, n_estimators=50, max_depth=10, min_samples_split=5):
    """
    Entrena un modelo Random Forest (RF).
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
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

def export_to_c(model, filename="rf_model.h"):
    """
    Exporta el modelo RF a código C++ utilizando micromlgen.
    """
    c_code = micromlgen.port(model)
    with open(filename, "w") as f:
        f.write(c_code)
    return c_code
