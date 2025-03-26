import logging
import os

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import KFold, GridSearchCV

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU cores, e.g., '4' for quad-core

app = Flask(__name__)

logging.basicConfig(filename='ml_insight_lab.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_data(X, y=None, normalize=False):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if normalize else X
    return X_scaled, y


def run_cross_validation(X, y, algo, params, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = {'mse': [], 'r2': [], 'accuracy': []}
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if algo == 'linear':
            model = LinearRegression()
        elif algo == 'logistic':
            model = LogisticRegression(C=1 / params.get('lr', 0.01), max_iter=100)
        elif algo == 'dtree':
            model = DecisionTreeClassifier(max_depth=params.get('depth', 3)) if params.get('is_classification',
                                                                                           False) else DecisionTreeRegressor(
                max_depth=params.get('depth', 3))
        elif algo == 'rf':
            model = RandomForestClassifier(max_depth=params.get('depth', 3), n_estimators=10) if params.get(
                'is_classification', False) else RandomForestRegressor(max_depth=params.get('depth', 3),
                                                                       n_estimators=10)
        elif algo == 'svm':
            model = SVC(kernel='rbf') if params.get('is_classification', False) else SVR(kernel='rbf')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if algo in ['linear', 'dtree', 'rf', 'svm'] and not params.get('is_classification', False):
            scores['mse'].append(mean_squared_error(y_test, y_pred))
            scores['r2'].append(r2_score(y_test, y_pred))
        else:
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
    return {k: np.mean(v) for k, v in scores.items() if v}


def grid_search(X, y, algo, param_grid):
    if algo == 'linear':
        model = LinearRegression()
        param_grid = {}
    elif algo == 'logistic':
        model = LogisticRegression(max_iter=100)
    elif algo == 'dtree':
        model = DecisionTreeClassifier() if param_grid.get('is_classification', False) else DecisionTreeRegressor()
    elif algo == 'rf':
        model = RandomForestClassifier(n_estimators=10) if param_grid.get('is_classification',
                                                                          False) else RandomForestRegressor(
            n_estimators=10)
    elif algo == 'svm':
        model = SVC() if param_grid.get('is_classification', False) else SVR()
    elif algo == 'kmeans':
        model = KMeans()
    elif algo == 'dbscan':
        return {}, {}  # DBSCAN not supported for grid search here
    elif algo == 'pca':
        model = PCA()

    grid = GridSearchCV(model, param_grid, cv=5,
                        scoring='accuracy' if algo in ['logistic', 'dtree', 'rf', 'svm'] and param_grid.get(
                            'is_classification', False) else 'r2' if algo in ['linear', 'dtree', 'rf',
                                                                              'svm'] else 'neg_mean_squared_error')
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


def linear_regression(X, y, lr=0.01, cv=False):
    X, y = preprocess_data(X, y)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {'mse': mean_squared_error(y, y_pred), 'r2': r2_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X, y, 'linear', {'lr': lr})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Linear Regression: {metrics}")
    return [[float(model.coef_[0]), float(model.intercept_)]], metrics


def logistic_regression(X, y, lr=0.01, cv=False):
    X, y = preprocess_data(X, y)
    model = LogisticRegression(C=1 / lr, max_iter=100)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {'accuracy': accuracy_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X, y, 'logistic', {'lr': lr})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Logistic Regression: {metrics}")
    return [[float(model.coef_[0][0]), float(model.intercept_[0])]], metrics


def decision_tree(X, y, max_depth=3, is_classification=False, cv=False):
    X, y = preprocess_data(X, y)
    model = DecisionTreeClassifier(max_depth=max_depth) if is_classification else DecisionTreeRegressor(
        max_depth=max_depth)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y,
                                                                                                                y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X, y, 'dtree', {'depth': max_depth, 'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    boundaries = []
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    for x in x_range:
        for y in y_range:
            pred = model.predict([[x, y] + [0] * (X.shape[1] - 2)])[0]
            boundaries.append([float(x), float(y), float(pred)])
    logging.info(f"Decision Tree: {metrics}")
    return boundaries, metrics


def random_forest(X, y, max_depth=3, is_classification=False, cv=False):
    X, y = preprocess_data(X, y)
    model = RandomForestClassifier(max_depth=max_depth,
                                   n_estimators=10) if is_classification else RandomForestRegressor(max_depth=max_depth,
                                                                                                    n_estimators=10)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y,
                                                                                                                y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X, y, 'rf', {'depth': max_depth, 'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    boundaries = []
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    for x in x_range:
        for y in y_range:
            pred = model.predict([[x, y] + [0] * (X.shape[1] - 2)])[0]
            boundaries.append([float(x), float(y), float(pred)])
    logging.info(f"Random Forest: {metrics}")
    return boundaries, metrics


def svm(X, y, is_classification=False, cv=False):
    X, y = preprocess_data(X, y)
    model = SVC(kernel='rbf') if is_classification else SVR(kernel='rbf')
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y,
                                                                                                                y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X, y, 'svm', {'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    boundaries = []
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    for x in x_range:
        for y in y_range:
            pred = model.predict([[x, y] + [0] * (X.shape[1] - 2)])[0]
            boundaries.append([float(x), float(y), float(pred)])
    logging.info(f"SVM: {metrics}")
    return boundaries, metrics


def kmeans(X, k=2):
    X, _ = preprocess_data(X)
    model = KMeans(n_clusters=k)
    model.fit(X)
    silhouette = silhouette_score(X, model.labels_) if len(set(model.labels_)) > 1 else 0
    history = [model.cluster_centers_.tolist()]
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(model.labels_):
        clusters[label].append(X[i].tolist())
    logging.info(f"K-Means: Silhouette={silhouette}")
    return history, clusters, {'silhouette': silhouette}


def dbscan(X, eps=0.5, min_samples=5):
    X, _ = preprocess_data(X)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    silhouette = silhouette_score(X, model.labels_) if len(set(model.labels_)) > 1 else 0
    clusters = [[] for _ in range(len(set(model.labels_)) - (1 if -1 in model.labels_ else 0))]
    for i, label in enumerate(model.labels_):
        if label != -1:
            clusters[label].append(X[i].tolist())
    logging.info(f"DBSCAN: Silhouette={silhouette}")
    return [], clusters, {'silhouette': silhouette}


def pca(X, n_components=2):
    X, _ = preprocess_data(X)
    model = PCA(n_components=n_components)
    transformed = model.fit_transform(X)
    logging.info(f"PCA: Variance={model.explained_variance_ratio_.tolist()}")
    return transformed.tolist(), model.explained_variance_ratio_.tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fit', methods=['POST'])
def fit():
    try:
        data = request.json
        if not data or 'X' not in data or 'algo' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        X = np.array(data['X'], dtype=float)
        y = np.array(data.get('y', []), dtype=float) if data.get('y') else None
        labels = data.get('labels', [])
        algo = data['algo']
        normalize = data.get('normalize', False)
        cv = data.get('cv', False)

        if labels and algo in ['logistic', 'dtree', 'rf', 'svm'] and any(isinstance(l, str) for l in labels):
            le = LabelEncoder()
            y = le.fit_transform(labels)
            logging.info(f"Encoded categorical labels for {algo}")

        if algo == 'linear':
            history, metrics = linear_regression(X, y, cv=cv)
            return jsonify({'history': history, 'metrics': metrics})
        elif algo == 'logistic':
            history, metrics = logistic_regression(X, y, cv=cv)
            return jsonify({'history': history, 'metrics': metrics})
        elif algo == 'dtree':
            depth = int(data.get('depth', 3))
            is_classification = data.get('is_classification', False)
            boundaries, metrics = decision_tree(X, y, max_depth=depth, is_classification=is_classification, cv=cv)
            return jsonify({'boundaries': boundaries, 'metrics': metrics})
        elif algo == 'rf':
            depth = int(data.get('depth', 3))
            is_classification = data.get('is_classification', False)
            boundaries, metrics = random_forest(X, y, max_depth=depth, is_classification=is_classification, cv=cv)
            return jsonify({'boundaries': boundaries, 'metrics': metrics})
        elif algo == 'svm':
            is_classification = data.get('is_classification', False)
            boundaries, metrics = svm(X, y, is_classification=is_classification, cv=cv)
            return jsonify({'boundaries': boundaries, 'metrics': metrics})
        elif algo == 'kmeans':
            k = int(data.get('k', 2))
            history, clusters, metrics = kmeans(X, k)
            return jsonify({'history': history, 'clusters': clusters, 'metrics': metrics})
        elif algo == 'dbscan':
            eps = float(data.get('eps', 0.5))
            min_samples = int(data.get('min_samples', 5))
            history, clusters, metrics = dbscan(X, eps=eps, min_samples=min_samples)
            return jsonify({'history': history, 'clusters': clusters, 'metrics': metrics})
        elif algo == 'pca':
            n_components = int(data.get('n_components', 2))
            transformed, variance = pca(X, n_components=n_components)
            return jsonify({'transformed': transformed, 'variance': variance})
        return jsonify({'error': 'Invalid algorithm'}), 400
    except ValueError as e:
        logging.error(f"Fit endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error in fit: {str(e)}")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500


@app.route('/grid_search', methods=['POST'])
def grid_search_endpoint():
    try:
        data = request.json
        X = np.array(data['X'], dtype=float)
        y = np.array(data.get('y', []), dtype=float) if data.get('y') else None
        algo = data['algo']
        param_grid = data['param_grid']

        if labels := data.get('labels', []):
            if algo in ['logistic', 'dtree', 'rf', 'svm'] and any(isinstance(l, str) for l in labels):
                le = LabelEncoder()
                y = le.fit_transform(labels)

        best_params, best_score = grid_search(X, y, algo, param_grid)
        return jsonify({'best_params': best_params, 'best_score': best_score})
    except Exception as e:
        logging.error(f"Grid search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/scenario')
def scenario():
    try:
        df = pd.read_csv('static/iris.csv')
        data = df.to_dict('records')[:20]
        logging.info("Iris scenario loaded")
        return jsonify(data)
    except FileNotFoundError:
        logging.error("Iris dataset not found")
        return jsonify({'error': 'Iris dataset not found'}), 500
    except Exception as e:
        logging.error(f"Scenario loading failed: {str(e)}")
        return jsonify({'error': f"Scenario loading failed: {str(e)}"}), 500


if __name__ == '__main__':
    logging.info("Starting ML Insight Lab application")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)