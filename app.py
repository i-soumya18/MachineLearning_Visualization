import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import logging
import os

# Configure logging
logging.basicConfig(filename='ml_insight_lab.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU cores

# Helper functions
def preprocess_data(X, y=None, normalize=False):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if normalize else X
    return X_scaled, y, scaler if normalize else None

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

# ML Model Functions
def linear_regression(X, y, lr=0.01, cv=False):
    X_proc, y, scaler = preprocess_data(X, y, normalize=True)
    model = LinearRegression()
    model.fit(X_proc, y)
    y_pred = model.predict(X_proc)
    metrics = {'mse': mean_squared_error(y, y_pred), 'r2': r2_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X_proc, y, 'linear', {'lr': lr})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Linear Regression: {metrics}")
    return model, metrics, scaler

def logistic_regression(X, y, lr=0.01, cv=False):
    X_proc, y, scaler = preprocess_data(X, y, normalize=True)
    model = LogisticRegression(C=1 / lr, max_iter=100)
    model.fit(X_proc, y)
    y_pred = model.predict(X_proc)
    metrics = {'accuracy': accuracy_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X_proc, y, 'logistic', {'lr': lr})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Logistic Regression: {metrics}")
    return model, metrics, scaler

def decision_tree(X, y, max_depth=3, is_classification=False, cv=False):
    X_proc, y, scaler = preprocess_data(X, y, normalize=True)
    model = DecisionTreeClassifier(max_depth=max_depth) if is_classification else DecisionTreeRegressor(
        max_depth=max_depth)
    model.fit(X_proc, y)
    y_pred = model.predict(X_proc)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X_proc, y, 'dtree', {'depth': max_depth, 'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Decision Tree: {metrics}")
    return model, metrics, scaler

def random_forest(X, y, max_depth=3, is_classification=False, cv=False):
    X_proc, y, scaler = preprocess_data(X, y, normalize=True)
    model = RandomForestClassifier(max_depth=max_depth,
                                   n_estimators=10) if is_classification else RandomForestRegressor(max_depth=max_depth,
                                                                                                    n_estimators=10)
    model.fit(X_proc, y)
    y_pred = model.predict(X_proc)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X_proc, y, 'rf', {'depth': max_depth, 'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"Random Forest: {metrics}")
    return model, metrics, scaler

def svm_model(X, y, is_classification=False, cv=False):
    X_proc, y, scaler = preprocess_data(X, y, normalize=True)
    model = SVC(kernel='rbf', probability=True) if is_classification else SVR(kernel='rbf')
    model.fit(X_proc, y)
    y_pred = model.predict(X_proc)
    metrics = {
        'accuracy' if is_classification else 'r2': accuracy_score(y, y_pred) if is_classification else r2_score(y, y_pred)}
    if cv:
        cv_metrics = run_cross_validation(X_proc, y, 'svm', {'is_classification': is_classification})
        metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
    logging.info(f"SVM: {metrics}")
    return model, metrics, scaler

def kmeans_model(X, k=2):
    X_proc, _, scaler = preprocess_data(X, normalize=True)
    model = KMeans(n_clusters=k)
    model.fit(X_proc)
    silhouette = silhouette_score(X_proc, model.labels_) if len(set(model.labels_)) > 1 else 0
    metrics = {'silhouette': silhouette}
    logging.info(f"K-Means: Silhouette={silhouette}")
    return model, metrics, scaler

def dbscan_model(X, eps=0.5, min_samples=5):
    X_proc, _, scaler = preprocess_data(X, normalize=True)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X_proc)
    silhouette = silhouette_score(X_proc, model.labels_) if len(set(model.labels_)) > 1 and len(set(model.labels_)) < len(X_proc) else 0
    metrics = {'silhouette': silhouette}
    logging.info(f"DBSCAN: Silhouette={silhouette}")
    return model, metrics, scaler

def pca_model(X, n_components=2):
    X_proc, _, scaler = preprocess_data(X, normalize=True)
    model = PCA(n_components=n_components)
    model.fit(X_proc)
    metrics = {'explained_variance': model.explained_variance_ratio_.tolist()}
    logging.info(f"PCA: Variance={model.explained_variance_ratio_.tolist()}")
    return model, metrics, scaler

def make_prediction(model, input_data, scaler=None, algo_type=None, is_classification=False):
    input_data = np.array(input_data).reshape(1, -1)
    if scaler:
        input_data = scaler.transform(input_data)
    
    if algo_type in ['kmeans']:
        prediction = model.predict(input_data)[0]
        return f"Cluster: {prediction}"
    elif is_classification:
        # Get prediction and probability for classification models
        prediction = model.predict(input_data)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)[0]
            prob_result = {f"Class {i}": f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
            return f"Predicted Class: {prediction}", prob_result
        return f"Predicted Class: {prediction}"
    else:
        # Regression prediction
        prediction = model.predict(input_data)[0]
        return f"Predicted Value: {prediction:.4f}"

# UI Function to create plots
def create_plot(X, y, model, algo, is_classification=False, scaler=None):
    if X.shape[1] < 2:
        return None  # Need at least 2 features for visualization
    
    fig = None
    
    # For supervised learning with 2D data
    if algo in ['linear', 'logistic', 'dtree', 'rf', 'svm'] and y is not None:
        if is_classification:
            # Create a scatter plot with decision boundaries for classification
            # Generate mesh grid for decision boundary
            X_proc = scaler.transform(X) if scaler else X
            x_min, x_max = X_proc[:, 0].min() - 1, X_proc[:, 0].max() + 1
            y_min, y_max = X_proc[:, 1].min() - 1, X_proc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig = px.scatter(x=X[:, 0], y=X[:, 1], color=[str(label) for label in y], 
                            labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Class'},
                            title=f"{algo.capitalize()} Classification")
            
            fig.add_trace(
                go.Contour(
                    z=Z,
                    x=np.arange(x_min, x_max, 0.02),
                    y=np.arange(y_min, y_max, 0.02),
                    colorscale='Viridis',
                    opacity=0.3,
                    showscale=False
                )
            )
        else:
            # Regression visualization
            fig = px.scatter(x=X[:, 0], y=y, 
                            labels={'x': 'Feature', 'y': 'Target'},
                            title=f"{algo.capitalize()} Regression")
            
            # Add regression line or curve
            X_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
            if scaler:
                X_range_scaled = scaler.transform(np.hstack([X_range, np.zeros((100, X.shape[1]-1))]))[:, 0].reshape(-1, 1)
                X_predict = np.hstack([X_range_scaled, np.zeros((100, X.shape[1]-1))])
            else:
                X_predict = np.hstack([X_range, np.zeros((100, X.shape[1]-1))])
            y_pred = model.predict(X_predict)
            
            fig.add_trace(
                go.Scatter(x=X_range.flatten(), y=y_pred, mode='lines', name='Model prediction', line=dict(color='red'))
            )
    
    # For unsupervised learning
    elif algo == 'kmeans':
        # K-means clustering visualization
        labels = model.labels_
        centers = model.cluster_centers_
        
        if scaler:
            # Transform centers back to original scale for plotting
            centers_orig = scaler.inverse_transform(centers)
        else:
            centers_orig = centers
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=[str(label) for label in labels],
                       labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Cluster'},
                       title="K-Means Clustering")
        
        # Add cluster centers
        for i, center in enumerate(centers_orig):
            fig.add_trace(
                go.Scatter(x=[center[0]], y=[center[1]], mode='markers', 
                           marker=dict(color='black', size=15, symbol='x'),
                           name=f'Cluster {i} center')
            )
    
    elif algo == 'dbscan':
        # DBSCAN clustering visualization
        labels = model.labels_
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=[str(label) for label in labels],
                       labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'Cluster'},
                       title="DBSCAN Clustering")
    
    elif algo == 'pca':
        # PCA visualization
        transformed = model.transform(scaler.transform(X) if scaler else X)
        var_explained = model.explained_variance_ratio_
        
        if transformed.shape[1] >= 2:
            fig = px.scatter(x=transformed[:, 0], y=transformed[:, 1],
                           labels={'x': f'PC1 ({var_explained[0]:.2%})', 
                                   'y': f'PC2 ({var_explained[1]:.2%})'},
                           title="PCA Projection")
    
    return fig

# Main UI
def main():
    st.set_page_config(page_title="ML Insight Lab", layout="wide", page_icon="📊")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("ML Insight Lab")
        st.subheader("Machine Learning Explorer")
        
        # Algorithm selection
        st.header("Algorithm Selection")
        algo_category = st.selectbox("Algorithm Category", 
                                     ["Supervised Learning", "Unsupervised Learning"])
        
        if algo_category == "Supervised Learning":
            algo = st.selectbox("Select Algorithm", 
                               ["Linear Regression", "Logistic Regression", "Decision Tree", 
                                "Random Forest", "SVM"])
        else:
            algo = st.selectbox("Select Algorithm", 
                               ["K-Means", "DBSCAN", "PCA"])
        
        # Data options
        st.header("Data Options")
        data_source = st.radio("Data Source", ["Upload CSV", "Use Iris Dataset", "Generate Random Data"])
        
        # Feature configuration
        st.header("Feature Configuration")
        normalize_data = st.checkbox("Normalize Data", value=True)
        use_cv = st.checkbox("Use Cross-Validation")
        
        # Add predict button in sidebar
        st.header("Prediction")
        st.write("Configure your model and then use the prediction form below")

    # Main content
    st.title("ML Insight Lab - Interactive ML Explorer")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = []
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
        
    # Data Loading Section
    st.header("1. Data Preparation")
    
    # Handle data loading based on selection
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.write("Data Preview:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif data_source == "Use Iris Dataset":
        try:
            df = pd.read_csv('static/Iris.csv')
            st.session_state.data = df
            st.write("Iris Dataset Preview:")
            st.dataframe(df.head())
        except FileNotFoundError:
            st.error("Iris dataset not found. Please ensure the file exists in the static directory.")
    
    elif data_source == "Generate Random Data":
        num_samples = st.slider("Number of samples", 10, 1000, 100)
        if algo_category == "Supervised Learning":
            if algo == "Linear Regression":
                # Generate linear data with noise
                X = np.random.rand(num_samples, 2) * 10
                y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(num_samples) * 2
                df = pd.DataFrame(np.column_stack([X, y]), columns=['feature1', 'feature2', 'target'])
            else:
                # Generate classification data
                X = np.random.rand(num_samples, 2) * 10
                y = (X[:, 0] + X[:, 1] > 10).astype(int)
                df = pd.DataFrame(np.column_stack([X, y]), columns=['feature1', 'feature2', 'target'])
        else:
            # Generate clustered data for unsupervised learning
            centers = [[2, 2], [8, 8], [2, 8], [8, 2]]
            cluster_std = 1.0
            n_per_cluster = num_samples // 4
            X = np.vstack([
                np.random.normal(centers[0], cluster_std, (n_per_cluster, 2)),
                np.random.normal(centers[1], cluster_std, (n_per_cluster, 2)),
                np.random.normal(centers[2], cluster_std, (n_per_cluster, 2)),
                np.random.normal(centers[3], cluster_std, (n_per_cluster, 2))
            ])
            df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        
        st.session_state.data = df
        st.write("Generated Data Preview:")
        st.dataframe(df.head())
    
    # Feature selection (if data is loaded)
    if st.session_state.data is not None:
        st.header("2. Feature Selection")
        
        cols = st.session_state.data.columns.tolist()
        
        # Skip feature selection for unsupervised learning algorithms
        if algo_category == "Supervised Learning":
            feature_cols = st.multiselect("Select feature columns", cols, 
                                         default=cols[:-1] if len(cols) > 1 else cols)
            target_col = st.selectbox("Select target column", cols, 
                                    index=len(cols)-1 if len(cols) > 1 else 0)
            
            if feature_cols and target_col:
                st.session_state.feature_columns = feature_cols
                st.session_state.target_column = target_col
                
                # Extract features and target
                X = st.session_state.data[feature_cols].values
                y = st.session_state.data[target_col].values
                
                # Handle categorical target for classification
                is_classification = False
                if algo in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]:
                    if pd.api.types.is_string_dtype(st.session_state.data[target_col]):
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        is_classification = True
                    elif len(np.unique(y)) < 10:  # Assume classification if fewer than 10 unique values
                        is_classification = True
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.is_classification = is_classification
        else:
            # For unsupervised learning, all columns are features
            feature_cols = st.multiselect("Select feature columns", cols, default=cols)
            if feature_cols:
                st.session_state.feature_columns = feature_cols
                st.session_state.X = st.session_state.data[feature_cols].values
    
        # Algorithm parameters configuration
        st.header("3. Algorithm Configuration")
        
        algorithm_params = {}
        
        # Show different parameters based on algorithm
        if algo == "Linear Regression":
            # Linear regression doesn't have many parameters to tune in sklearn
            pass
        
        elif algo == "Logistic Regression":
            lr_value = st.slider("Learning Rate (1/C)", 0.001, 1.0, 0.01, 0.001)
            algorithm_params['lr'] = lr_value
        
        elif algo == "Decision Tree" or algo == "Random Forest":
            max_depth = st.slider("Max Depth", 1, 10, 3)
            algorithm_params['depth'] = max_depth
            algorithm_params['is_classification'] = st.session_state.get('is_classification', False)
        
        elif algo == "SVM":
            algorithm_params['is_classification'] = st.session_state.get('is_classification', False)
        
        elif algo == "K-Means":
            k_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
            algorithm_params['k'] = k_clusters
        
        elif algo == "DBSCAN":
            eps_value = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            algorithm_params['eps'] = eps_value
            algorithm_params['min_samples'] = min_samples
        
        elif algo == "PCA":
            n_components = st.slider("Number of Components", 1, 
                                   min(5, len(st.session_state.feature_columns) if st.session_state.feature_columns else 2), 2)
            algorithm_params['n_components'] = n_components
        
        # Run Algorithm button
        if st.button("Run Algorithm"):
            if st.session_state.X is not None:
                with st.spinner("Running algorithm..."):
                    # Run the selected algorithm
                    try:
                        if algo == "Linear Regression":
                            model, metrics, scaler = linear_regression(
                                st.session_state.X, st.session_state.y, 
                                cv=use_cv)
                            
                        elif algo == "Logistic Regression":
                            model, metrics, scaler = logistic_regression(
                                st.session_state.X, st.session_state.y, 
                                lr=algorithm_params.get('lr', 0.01), 
                                cv=use_cv)
                            
                        elif algo == "Decision Tree":
                            model, metrics, scaler = decision_tree(
                                st.session_state.X, st.session_state.y, 
                                max_depth=algorithm_params.get('depth', 3), 
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "Random Forest":
                            model, metrics, scaler = random_forest(
                                st.session_state.X, st.session_state.y, 
                                max_depth=algorithm_params.get('depth', 3), 
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "SVM":
                            model, metrics, scaler = svm_model(
                                st.session_state.X, st.session_state.y, 
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "K-Means":
                            model, metrics, scaler = kmeans_model(
                                st.session_state.X, 
                                k=algorithm_params.get('k', 3))
                            
                        elif algo == "DBSCAN":
                            model, metrics, scaler = dbscan_model(
                                st.session_state.X, 
                                eps=algorithm_params.get('eps', 0.5), 
                                min_samples=algorithm_params.get('min_samples', 5))
                            
                        elif algo == "PCA":
                            model, metrics, scaler = pca_model(
                                st.session_state.X, 
                                n_components=algorithm_params.get('n_components', 2))
                        
                        # Store model and metrics in session state
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        st.session_state.scaler = scaler
                        st.session_state.algo = algo.lower().replace(" ", "_")
                        st.session_state.is_fitted = True
                        
                        st.success(f"{algo} completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error running algorithm: {str(e)}")
                        logging.error(f"Algorithm error: {str(e)}")
            else:
                st.warning("Please prepare your data first!")
                
        # Results section
        if st.session_state.get('is_fitted', False):
            st.header("4. Results and Visualization")
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame(st.session_state.metrics.items(), columns=['Metric', 'Value'])
            st.table(metrics_df)
            
            # Create and show visualization
            if st.session_state.X is not None and st.session_state.X.shape[1] >= 2:
                fig = create_plot(
                    st.session_state.X, 
                    st.session_state.get('y'), 
                    st.session_state.model,
                    st.session_state.algo,
                    is_classification=st.session_state.get('is_classification', False),
                    scaler=st.session_state.scaler
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Visualization is not available for this configuration.")
            else:
                st.info("At least 2 features are required for visualization.")
            
            # Prediction section
            st.header("5. Make Predictions")
            
            if st.session_state.feature_columns:
                st.subheader("Enter Values for Prediction")
                
                # Create input fields for each feature
                input_values = []
                for feature in st.session_state.feature_columns:
                    input_val = st.number_input(f"Enter value for {feature}", 
                                              value=float(st.session_state.X[:, st.session_state.feature_columns.index(feature)].mean()))
                    input_values.append(input_val)
                
                # Prediction button
                if st.button("Predict"):
                    try:
                        prediction_result = make_prediction(
                            st.session_state.model,
                            input_values,
                            scaler=st.session_state.scaler,
                            algo_type=st.session_state.algo,
                            is_classification=st.session_state.get('is_classification', False)
                        )
                        
                        st.success("Prediction completed!")
                        
                        # Display prediction result
                        if isinstance(prediction_result, tuple):
                            # For classification with probabilities
                            pred_label, pred_probs = prediction_result
                            st.write(f"### {pred_label}")
                            st.write("#### Class Probabilities:")
                            for class_label, prob in pred_probs.items():
                                st.write(f"{class_label}: {prob}")
                        else:
                            # For regression or simple classification
                            st.write(f"### {prediction_result}")
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        logging.error(f"Prediction error: {str(e)}")

    # Footer
    st.markdown("---")
    st.write("ML Insight Lab © 2023 - Advanced Machine Learning Exploration & Visualization")

if __name__ == "__main__":
    main()