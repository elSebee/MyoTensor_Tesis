from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import micromlgen

def train(X_train, y_train, C=1000, gamma='scale', kernel='rbf'):
    """
    Entrena un modelo Support Vector Machine (SVM).
    """
    model = SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_with_gridsearch(X_train, y_train):
    """
    Optimiza y entrena un modelo SVM usando GridSearchCV.
    """
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'sigmoid']
    }
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

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
