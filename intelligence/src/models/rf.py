from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

def train_with_gridsearch(X_train, y_train):
    """
    Optimiza y entrena un modelo Random Forest usando GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf_base,
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

def export_to_c(model, filename="rf_model.h"):
    """
    Exporta el modelo RF a código C++ utilizando micromlgen.
    """
    c_code = micromlgen.port(model)
    with open(filename, "w") as f:
        f.write(c_code)
    return c_code
