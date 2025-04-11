import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, silhouette_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import logging
import traceback
import os
import uuid
import datetime
import atexit
import shutil

# Configure logging
logging.basicConfig(filename='ml_insight_lab.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU cores

# Create temp directory if it doesn't exist
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to clean up temporary files
def cleanup_temp_files(max_age_hours=24):
    """Delete temporary files older than max_age_hours"""
    try:
        current_time = datetime.datetime.now()
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if (current_time - file_creation_time).total_seconds() > max_age_hours * 3600:
                    os.remove(file_path)
                    logging.info(f"Removed old temporary file: {file_path}")
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")

# Register the cleanup function to run on exit
atexit.register(cleanup_temp_files)

# Class to handle exceptions and provide user-friendly messages
class MLInsightException(Exception):
    """Custom exception class for ML Insight Lab with user-friendly messages"""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)
        
        # Log the error
        if details:
            logging.error(f"{message}: {details}")
            logging.error(traceback.format_exc())
        else:
            logging.error(message)
            logging.error(traceback.format_exc())

# Helper functions for data handling
def detect_data_types(df):
    """Detect and categorize columns by data type"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'date': date_cols
    }

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to temp directory and return the file path"""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file_name = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(TEMP_DIR, temp_file_name)
        
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logging.info(f"Saved uploaded file to {temp_file_path}")
        return temp_file_path
    except Exception as e:
        raise MLInsightException("Failed to save uploaded file", str(e))

def load_dataset(file_path):
    """Load dataset from file with type detection"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise MLInsightException(f"Unsupported file format: {file_ext}")
        
        return df
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Failed to load dataset", str(e))

# Helper functions
def preprocess_data(X, y=None, categorical_features=None, numeric_features=None, normalize=False):
    """
    Preprocess data with handling for categorical features
    Returns preprocessed data and preprocessing objects that can be used for new data
    """
    try:
        preprocessors = {}
        
        # Handle empty or None inputs
        if X is None or X.size == 0:
            raise MLInsightException("No data provided for preprocessing")
        
        # Create copies to avoid modifying original data
        X_processed = X.copy()
        
        # Process categorical features if any
        if categorical_features and len(categorical_features) > 0:
            # For each categorical feature, create a label encoder
            label_encoders = {}
            for cat_col in categorical_features:
                if cat_col in X_processed.columns:
                    le = LabelEncoder()
                    X_processed[cat_col] = le.fit_transform(X_processed[cat_col].astype(str))
                    label_encoders[cat_col] = le
            
            preprocessors['label_encoders'] = label_encoders
        
        # Normalize numeric features if requested
        if normalize and numeric_features and len(numeric_features) > 0:
            scaler = StandardScaler()
            numeric_data = X_processed[numeric_features].values
            X_processed[numeric_features] = scaler.fit_transform(numeric_data)
            preprocessors['scaler'] = scaler
        
        return X_processed, y, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Data preprocessing failed", str(e))

def run_cross_validation(X, y, algo, params, k_folds=5):
    """Run cross-validation with enhanced error handling"""
    try:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = {'mse': [], 'r2': [], 'accuracy': [], 'confusion_matrices': []}
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Select model based on algorithm
            if algo == 'linear':
                model = LinearRegression()
            elif algo == 'logistic':
                model = LogisticRegression(C=1/params.get('lr', 0.01), max_iter=1000, solver='liblinear')
            elif algo == 'dtree':
                if params.get('is_classification', False):
                    model = DecisionTreeClassifier(max_depth=params.get('depth', 3), random_state=42)
                else:
                    model = DecisionTreeRegressor(max_depth=params.get('depth', 3), random_state=42)
            elif algo == 'rf':
                if params.get('is_classification', False):
                    model = RandomForestClassifier(
                        max_depth=params.get('depth', 3), 
                        n_estimators=params.get('n_estimators', 10),
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        max_depth=params.get('depth', 3), 
                        n_estimators=params.get('n_estimators', 10),
                        random_state=42
                    )
            elif algo == 'svm':
                if params.get('is_classification', False):
                    model = SVC(kernel=params.get('kernel', 'rbf'), probability=True, random_state=42)
                else:
                    model = SVR(kernel=params.get('kernel', 'rbf'))
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if algo in ['linear', 'dtree', 'rf', 'svm'] and not params.get('is_classification', False):
                scores['mse'].append(mean_squared_error(y_test, y_pred))
                scores['r2'].append(r2_score(y_test, y_pred))
            else:
                scores['accuracy'].append(accuracy_score(y_test, y_pred))
                scores['confusion_matrices'].append(confusion_matrix(y_test, y_pred))
                
        # Compute average scores
        result = {}
        for k, v in scores.items():
            if k != 'confusion_matrices':
                result[k] = np.mean(v) if v else None
        
        # For classification, add the average confusion matrix
        if 'confusion_matrices' in scores and scores['confusion_matrices']:
            matrices = scores['confusion_matrices']
            if matrices and len(matrices) > 0:
                # Get the shape of the first matrix to initialize the average
                shape = matrices[0].shape
                avg_matrix = np.zeros(shape)
                
                for matrix in matrices:
                    avg_matrix += matrix
                
                avg_matrix = avg_matrix / len(matrices)
                result['avg_confusion_matrix'] = avg_matrix
        
        return result
    except Exception as e:
        raise MLInsightException("Cross-validation failed", str(e))

# ML Model Functions with error handling
def linear_regression(X, y, cv=False):
    """Linear regression with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        data_types = detect_data_types(X)
        X_proc, y, preprocessors = preprocess_data(
            X, y, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        model = LinearRegression()
        model.fit(X_proc, y)
        y_pred = model.predict(X_proc)
        metrics = {'mse': mean_squared_error(y, y_pred), 'r2': r2_score(y, y_pred)}
        
        if cv:
            cv_metrics = run_cross_validation(X_proc, y, 'linear', {})
            metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
            
        logging.info(f"Linear Regression: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Linear regression failed", str(e))

def logistic_regression(X, y, lr=0.01, cv=False):
    """Logistic regression with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, y, preprocessors = preprocess_data(
            X, y, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        model = LogisticRegression(C=1/lr, max_iter=1000, solver='liblinear', random_state=42)
        model.fit(X_proc, y)
        y_pred = model.predict(X_proc)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        if cv:
            cv_metrics = run_cross_validation(X_proc, y, 'logistic', {'lr': lr})
            metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
            if 'cv_avg_confusion_matrix' in metrics:
                metrics['cv_avg_confusion_matrix'] = metrics['cv_avg_confusion_matrix'].tolist()
                
        logging.info(f"Logistic Regression: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Logistic regression failed", str(e))

def decision_tree(X, y, max_depth=3, is_classification=False, cv=False):
    """Decision tree with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, y, preprocessors = preprocess_data(
            X, y, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        # Select model based on problem type
        if is_classification:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            
        model.fit(X_proc, y)
        y_pred = model.predict(X_proc)
        
        # Calculate metrics based on problem type
        if is_classification:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        # Run cross-validation if requested
        if cv:
            cv_metrics = run_cross_validation(
                X_proc, y, 'dtree', 
                {'depth': max_depth, 'is_classification': is_classification}
            )
            metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
            if 'cv_avg_confusion_matrix' in metrics:
                metrics['cv_avg_confusion_matrix'] = metrics['cv_avg_confusion_matrix'].tolist()
                
        logging.info(f"Decision Tree: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Decision tree failed", str(e))

def random_forest(X, y, max_depth=3, n_estimators=10, is_classification=False, cv=False):
    """Random forest with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, y, preprocessors = preprocess_data(
            X, y, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        # Select model based on problem type
        if is_classification:
            model = RandomForestClassifier(
                max_depth=max_depth, 
                n_estimators=n_estimators,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                max_depth=max_depth, 
                n_estimators=n_estimators,
                random_state=42
            )
            
        model.fit(X_proc, y)
        y_pred = model.predict(X_proc)
        
        # Calculate metrics based on problem type
        if is_classification:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        # Run cross-validation if requested
        if cv:
            cv_metrics = run_cross_validation(
                X_proc, y, 'rf', 
                {
                    'depth': max_depth, 
                    'n_estimators': n_estimators,
                    'is_classification': is_classification
                }
            )
            metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
            if 'cv_avg_confusion_matrix' in metrics:
                metrics['cv_avg_confusion_matrix'] = metrics['cv_avg_confusion_matrix'].tolist()
                
        logging.info(f"Random Forest: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("Random forest failed", str(e))

def svm_model(X, y, kernel='rbf', is_classification=False, cv=False):
    """SVM with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, y, preprocessors = preprocess_data(
            X, y, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        # Select model based on problem type
        if is_classification:
            model = SVC(kernel=kernel, probability=True, random_state=42)
        else:
            model = SVR(kernel=kernel)
            
        model.fit(X_proc, y)
        y_pred = model.predict(X_proc)
        
        # Calculate metrics based on problem type
        if is_classification:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        # Run cross-validation if requested
        if cv:
            cv_metrics = run_cross_validation(
                X_proc, y, 'svm', 
                {'kernel': kernel, 'is_classification': is_classification}
            )
            metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
            if 'cv_avg_confusion_matrix' in metrics:
                metrics['cv_avg_confusion_matrix'] = metrics['cv_avg_confusion_matrix'].tolist()
                
        logging.info(f"SVM: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("SVM failed", str(e))

def kmeans_model(X, k=3):
    """K-means clustering with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, _, preprocessors = preprocess_data(
            X, None, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_proc)
        
        # Calculate silhouette score if possible
        silhouette = 0
        if len(set(model.labels_)) > 1 and len(set(model.labels_)) < len(X_proc):
            silhouette = silhouette_score(X_proc, model.labels_)
        
        metrics = {
            'silhouette': silhouette,
            'inertia': model.inertia_,
            'n_clusters': k
        }
                
        logging.info(f"K-Means: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("K-means clustering failed", str(e))

def dbscan_model(X, eps=0.5, min_samples=5):
    """DBSCAN clustering with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, _, preprocessors = preprocess_data(
            X, None, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X_proc)
        
        # Calculate silhouette score if possible
        silhouette = 0
        if len(set(model.labels_)) > 1 and len(set(model.labels_)) < len(X_proc) and -1 not in model.labels_:
            silhouette = silhouette_score(X_proc, model.labels_)
        
        metrics = {
            'silhouette': silhouette,
            'n_clusters': len(set(model.labels_)) - (1 if -1 in model.labels_ else 0),
            'n_noise': list(model.labels_).count(-1)
        }
                
        logging.info(f"DBSCAN: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("DBSCAN clustering failed", str(e))

def pca_model(X, n_components=2):
    """PCA dimensionality reduction with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        data_types = detect_data_types(X)
        X_proc, _, preprocessors = preprocess_data(
            X, None, 
            categorical_features=data_types['categorical'],
            numeric_features=data_types['numeric'],
            normalize=True
        )
        
        # Check if n_components is valid
        n_components = min(n_components, X_proc.shape[1])
        
        model = PCA(n_components=n_components, random_state=42)
        model.fit(X_proc)
        
        metrics = {
            'explained_variance': model.explained_variance_ratio_.tolist(),
            'total_explained_variance': np.sum(model.explained_variance_ratio_),
            'n_components': n_components
        }
                
        logging.info(f"PCA: {metrics}")
        return model, metrics, preprocessors
    except Exception as e:
        if isinstance(e, MLInsightException):
            raise e
        else:
            raise MLInsightException("PCA failed", str(e))

def make_prediction(model, input_data, preprocessors=None, algo_type=None, is_classification=False):
    """Make prediction with error handling and input processing"""
    try:
        # Convert input to DataFrame if it's not already
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
        
        # Apply preprocessing if provided
        if preprocessors:
            # Apply label encoding to categorical features
            if 'label_encoders' in preprocessors:
                for col, le in preprocessors['label_encoders'].items():
                    if col in input_data.columns:
                        input_data[col] = le.transform(input_data[col].astype(str))
            
            # Apply scaling to numeric features
            if 'scaler' in preprocessors:
                numeric_cols = [col for col in input_data.columns if col in preprocessors.get('numeric_features', [])]
                if numeric_cols:
                    input_data[numeric_cols] = preprocessors['scaler'].transform(input_data[numeric_cols])
        
        # Make prediction based on algorithm type
        if algo_type in ['kmeans']:
            prediction = model.predict(input_data)[0]
            return f"Cluster: {prediction}"
        elif is_classification:
            # Get prediction and probability for classification models
            prediction = model.predict(input_data)[0]
            result = {
                'class': str(prediction)
            }
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)[0]
                result['probabilities'] = {
                    f"Class {i}": f"{prob*100:.2f}%" 
                    for i, prob in enumerate(probabilities)
                }
            return result
        else:
            # Regression prediction
            prediction = model.predict(input_data)[0]
            return {'value': float(prediction)}
    except Exception as e:
        raise MLInsightException("Prediction failed", str(e))

# UI Function to create plots
def create_plot(X, y, model, algo, is_classification=False, preprocessors=None):
    """Create visualizations with error handling"""
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if X.shape[1] < 2:
            return None  # Need at least 2 features for visualization
        
        fig = None
        
        # For supervised learning with 2D data
        if algo in ['linear', 'logistic', 'dtree', 'rf', 'svm'] and y is not None:
            # Get the first two features for visualization
            features = X.columns[:2].tolist()
            X_2d = X[features]
            
            if is_classification:
                # Create a scatter plot with decision boundaries for classification
                # Generate mesh grid for decision boundary
                x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
                y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                    np.arange(y_min, y_max, 0.02))
                
                # Create mesh data for prediction
                mesh_data = pd.DataFrame({
                    features[0]: xx.ravel(),
                    features[1]: yy.ravel()
                })
                
                # Fill in other features with mean values
                for col in X.columns:
                    if col not in features:
                        mesh_data[col] = X[col].mean()
                
                # Apply preprocessing to mesh data
                if preprocessors:
                    if 'label_encoders' in preprocessors:
                        for col, le in preprocessors['label_encoders'].items():
                            if col in mesh_data.columns:
                                mesh_data[col] = le.transform(mesh_data[col].astype(str))
                
                # Make predictions for decision boundary
                Z = model.predict(mesh_data)
                Z = Z.reshape(xx.shape)
                
                fig = px.scatter(x=X_2d.iloc[:, 0], y=X_2d.iloc[:, 1], color=[str(label) for label in y], 
                                labels={'x': features[0], 'y': features[1], 'color': 'Class'},
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
                fig = px.scatter(x=X_2d.iloc[:, 0], y=y, 
                                labels={'x': features[0], 'y': 'Target'},
                                title=f"{algo.capitalize()} Regression")
                
                # Add regression line or curve
                x_range = np.linspace(X_2d.iloc[:, 0].min(), X_2d.iloc[:, 0].max(), 100)
                
                # Create prediction data
                pred_data = pd.DataFrame({features[0]: x_range})
                for col in X.columns:
                    if col != features[0]:
                        # Use mean values for other features
                        pred_data[col] = X[col].mean()
                
                # Apply preprocessing to prediction data
                if preprocessors:
                    if 'label_encoders' in preprocessors:
                        for col, le in preprocessors['label_encoders'].items():
                            if col in pred_data.columns:
                                pred_data[col] = le.transform(pred_data[col].astype(str))
                
                # Make predictions
                y_pred = model.predict(pred_data)
                
                fig.add_trace(
                    go.Scatter(x=x_range, y=y_pred, mode='lines', name='Model prediction', line=dict(color='red'))
                )
        
        # For unsupervised learning
        elif algo == 'kmeans':
            # K-means clustering visualization
            features = X.columns[:2].tolist()
            X_2d = X[features]
            
            labels = model.labels_
            centers = model.cluster_centers_
            
            fig = px.scatter(x=X_2d.iloc[:, 0], y=X_2d.iloc[:, 1], color=[str(label) for label in labels],
                           labels={'x': features[0], 'y': features[1], 'color': 'Cluster'},
                           title="K-Means Clustering")
            
            # Add cluster centers
            centers_2d = centers[:, :2]  # Get first two dimensions of centers
            
            for i, center in enumerate(centers_2d):
                fig.add_trace(
                    go.Scatter(x=[center[0]], y=[center[1]], mode='markers', 
                               marker=dict(color='black', size=15, symbol='x'),
                               name=f'Cluster {i} center')
                )
        
        elif algo == 'dbscan':
            # DBSCAN clustering visualization
            features = X.columns[:2].tolist()
            X_2d = X[features]
            
            labels = model.labels_
            
            fig = px.scatter(x=X_2d.iloc[:, 0], y=X_2d.iloc[:, 1], color=[str(label) for label in labels],
                           labels={'x': features[0], 'y': features[1], 'color': 'Cluster'},
                           title="DBSCAN Clustering")
        
        elif algo == 'pca':
            # PCA visualization
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
                
            data_types = detect_data_types(X)
            X_proc, _, _ = preprocess_data(
                X, None, 
                categorical_features=data_types['categorical'],
                numeric_features=data_types['numeric'],
                normalize=True
            )
            
            transformed = model.transform(X_proc)
            var_explained = model.explained_variance_ratio_
            
            if transformed.shape[1] >= 2:
                if y is not None:
                    # Colored by target if available
                    df_plot = pd.DataFrame({
                        'PC1': transformed[:, 0],
                        'PC2': transformed[:, 1],
                        'Target': [str(val) for val in y]
                    })
                    
                    fig = px.scatter(df_plot, x='PC1', y='PC2', color='Target',
                                   labels={'x': f'PC1 ({var_explained[0]:.2%})', 
                                           'y': f'PC2 ({var_explained[1]:.2%})'},
                                   title="PCA Projection")
                else:
                    # Without target coloring
                    df_plot = pd.DataFrame({
                        'PC1': transformed[:, 0],
                        'PC2': transformed[:, 1]
                    })
                    
                    fig = px.scatter(df_plot, x='PC1', y='PC2',
                                   labels={'x': f'PC1 ({var_explained[0]:.2%})', 
                                           'y': f'PC2 ({var_explained[1]:.2%})'},
                                   title="PCA Projection")
                
                # Add variance explained as text
                fig.add_annotation(
                    text=f"Total variance explained: {np.sum(var_explained):.2%}",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False
                )
        
        return fig
    except Exception as e:
        logging.error(f"Error creating plot: {str(e)}")
        logging.error(traceback.format_exc())
        return None

# Main UI
def main():
    st.set_page_config(page_title="ML Insight Lab", layout="wide", page_icon="📊")
    
    # Run cleanup on startup
    cleanup_temp_files()
    
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
        data_source = st.radio("Data Source", ["Upload File", "Use Iris Dataset", "Generate Random Data"])
        
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
    if 'preprocessors' not in st.session_state:
        st.session_state.preprocessors = None
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
    if 'temp_file_path' not in st.session_state:
        st.session_state.temp_file_path = None
    if 'data_types' not in st.session_state:
        st.session_state.data_types = None
        
    # Data Loading Section
    st.header("1. Data Preparation")
    
    # Handle data loading based on selection
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload a file (CSV, Excel, JSON)", 
                                        type=["csv", "xlsx", "xls", "json"])
        if uploaded_file is not None:
            try:
                # Save uploaded file to temp directory
                temp_file_path = save_uploaded_file(uploaded_file)
                st.session_state.temp_file_path = temp_file_path
                
                # Load the dataset
                df = load_dataset(temp_file_path)
                st.session_state.data = df
                
                # Detect data types
                st.session_state.data_types = detect_data_types(df)
                
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Show data type information
                st.write("Data Types:")
                data_types = st.session_state.data_types
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Numeric Features:", ", ".join(data_types['numeric']) if data_types['numeric'] else "None")
                with col2:
                    st.write("Categorical Features:", ", ".join(data_types['categorical']) if data_types['categorical'] else "None")
                
            except MLInsightException as e:
                st.error(f"Error: {e.message}")
                if e.details:
                    st.error(f"Details: {e.details}")
    
    elif data_source == "Use Iris Dataset":
        try:
            iris_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/Iris.csv')
            df = load_dataset(iris_path)
            st.session_state.data = df
            
            # Detect data types
            st.session_state.data_types = detect_data_types(df)
            
            st.write("Iris Dataset Preview:")
            st.dataframe(df.head())
            
            # Show data type information
            st.write("Data Types:")
            data_types = st.session_state.data_types
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numeric Features:", ", ".join(data_types['numeric']) if data_types['numeric'] else "None")
            with col2:
                st.write("Categorical Features:", ", ".join(data_types['categorical']) if data_types['categorical'] else "None")
                
        except MLInsightException as e:
            st.error(f"Error: {e.message}")
            if e.details:
                st.error(f"Details: {e.details}")
    
    elif data_source == "Generate Random Data":
        num_samples = st.slider("Number of samples", 10, 1000, 100)
        
        try:
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
            
            # Detect data types
            st.session_state.data_types = detect_data_types(df)
            
            st.write("Generated Data Preview:")
            st.dataframe(df.head())
            
            # Show data type information
            st.write("Data Types:")
            data_types = st.session_state.data_types
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numeric Features:", ", ".join(data_types['numeric']) if data_types['numeric'] else "None")
            with col2:
                st.write("Categorical Features:", ", ".join(data_types['categorical']) if data_types['categorical'] else "None")
                
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
            logging.error(f"Data generation error: {str(e)}")
    
    # Feature selection (if data is loaded)
    if st.session_state.data is not None:
        st.header("2. Feature Selection")
        
        cols = st.session_state.data.columns.tolist()
        
        # Skip feature selection for unsupervised learning algorithms
        if algo_category == "Supervised Learning":
            feature_cols = st.multiselect("Select feature columns", cols, 
                                         default=[col for col in cols if col != cols[-1]] if len(cols) > 1 else cols)
            target_col = st.selectbox("Select target column", cols, 
                                    index=len(cols)-1 if len(cols) > 1 else 0)
            
            if feature_cols and target_col:
                st.session_state.feature_columns = feature_cols
                st.session_state.target_column = target_col
                
                try:
                    # Extract features and target
                    X = st.session_state.data[feature_cols]
                    y = st.session_state.data[target_col].values
                    
                    # Handle categorical target for classification
                    is_classification = False
                    if algo in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]:
                        if pd.api.types.is_string_dtype(st.session_state.data[target_col]):
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            is_classification = True
                            st.info(f"Categorical target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
                        elif len(np.unique(y)) < 10:  # Assume classification if fewer than 10 unique values
                            is_classification = True
                    
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.is_classification = is_classification
                    
                except Exception as e:
                    st.error(f"Error preparing data: {str(e)}")
                    logging.error(f"Data preparation error: {str(e)}")
        else:
            # For unsupervised learning, all columns are features by default
            # But allow user to select which ones to use
            feature_cols = st.multiselect("Select feature columns", cols, default=cols)
            if feature_cols:
                st.session_state.feature_columns = feature_cols
                try:
                    st.session_state.X = st.session_state.data[feature_cols]
                except Exception as e:
                    st.error(f"Error selecting features: {str(e)}")
                    logging.error(f"Feature selection error: {str(e)}")
    
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
        
        elif algo == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 3)
            algorithm_params['depth'] = max_depth
            algorithm_params['is_classification'] = st.session_state.get('is_classification', False)
        
        elif algo == "Random Forest":
            max_depth = st.slider("Max Depth", 1, 20, 3)
            n_estimators = st.slider("Number of Trees", 10, 200, 50, 10)
            algorithm_params['depth'] = max_depth
            algorithm_params['n_estimators'] = n_estimators
            algorithm_params['is_classification'] = st.session_state.get('is_classification', False)
        
        elif algo == "SVM":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], 0)
            algorithm_params['kernel'] = kernel
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
            max_components = min(10, len(st.session_state.feature_columns) if st.session_state.feature_columns else 2)
            n_components = st.slider("Number of Components", 1, max_components, 2)
            algorithm_params['n_components'] = n_components
        
        # Run Algorithm button
        if st.button("Run Algorithm"):
            if st.session_state.X is not None:
                with st.spinner("Running algorithm..."):
                    # Run the selected algorithm
                    try:
                        if algo == "Linear Regression":
                            model, metrics, preprocessors = linear_regression(
                                st.session_state.X, st.session_state.y, 
                                cv=use_cv)
                            
                        elif algo == "Logistic Regression":
                            model, metrics, preprocessors = logistic_regression(
                                st.session_state.X, st.session_state.y, 
                                lr=algorithm_params.get('lr', 0.01), 
                                cv=use_cv)
                            
                        elif algo == "Decision Tree":
                            model, metrics, preprocessors = decision_tree(
                                st.session_state.X, st.session_state.y, 
                                max_depth=algorithm_params.get('depth', 3), 
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "Random Forest":
                            model, metrics, preprocessors = random_forest(
                                st.session_state.X, st.session_state.y, 
                                max_depth=algorithm_params.get('depth', 3),
                                n_estimators=algorithm_params.get('n_estimators', 10),
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "SVM":
                            model, metrics, preprocessors = svm_model(
                                st.session_state.X, st.session_state.y,
                                kernel=algorithm_params.get('kernel', 'rbf'),
                                is_classification=algorithm_params.get('is_classification', False),
                                cv=use_cv)
                            
                        elif algo == "K-Means":
                            model, metrics, preprocessors = kmeans_model(
                                st.session_state.X, 
                                k=algorithm_params.get('k', 3))
                            
                        elif algo == "DBSCAN":
                            model, metrics, preprocessors = dbscan_model(
                                st.session_state.X, 
                                eps=algorithm_params.get('eps', 0.5), 
                                min_samples=algorithm_params.get('min_samples', 5))
                            
                        elif algo == "PCA":
                            model, metrics, preprocessors = pca_model(
                                st.session_state.X, 
                                n_components=algorithm_params.get('n_components', 2))
                        
                        # Store model and metrics in session state
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        st.session_state.preprocessors = preprocessors
                        st.session_state.algo = algo.lower().replace(" ", "_")
                        st.session_state.is_fitted = True
                        st.session_state.algorithm_params = algorithm_params
                        
                        st.success(f"{algo} completed successfully!")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if isinstance(e, MLInsightException):
                            st.error(f"Error: {e.message}")
                            if e.details:
                                st.error(f"Details: {e.details}")
                        else:
                            st.error(f"Error running algorithm: {error_msg}")
                        logging.error(f"Algorithm error: {error_msg}")
                        logging.error(traceback.format_exc())
            else:
                st.warning("Please prepare your data first!")
                
        # Results section
        if st.session_state.get('is_fitted', False):
            st.header("4. Results and Visualization")
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            metrics = st.session_state.metrics
            
            # Format metrics for display
            display_metrics = {}
            for k, v in metrics.items():
                if k in ['confusion_matrix', 'classification_report', 'cv_avg_confusion_matrix']:
                    continue  # Display these separately
                if isinstance(v, float):
                    display_metrics[k] = f"{v:.4f}"
                else:
                    display_metrics[k] = v
            
            metrics_df = pd.DataFrame(display_metrics.items(), columns=['Metric', 'Value'])
            st.table(metrics_df)
            
            # Display confusion matrix if available
            if 'confusion_matrix' in metrics:
                st.subheader("Confusion Matrix")
                cm = np.array(metrics['confusion_matrix'])
                
                # Create a heatmap
                fig = px.imshow(cm, 
                              text_auto=True, 
                              color_continuous_scale='Blues',
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Display classification report if available
            if 'classification_report' in metrics and isinstance(metrics['classification_report'], dict):
                st.subheader("Classification Report")
                
                # Extract per-class metrics
                class_metrics = {}
                for class_name, class_metrics_dict in metrics['classification_report'].items():
                    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                        if isinstance(class_metrics_dict, dict):
                            class_metrics[class_name] = class_metrics_dict
                
                # Create DataFrame for display
                if class_metrics:
                    class_df = pd.DataFrame(class_metrics).T
                    class_df = class_df.reset_index().rename(columns={'index': 'Class'})
                    st.dataframe(class_df)
            
            # Create and show visualization
            if st.session_state.X is not None and st.session_state.X.shape[1] >= 2:
                st.subheader("Visualization")
                fig = create_plot(
                    st.session_state.X, 
                    st.session_state.get('y'), 
                    st.session_state.model,
                    st.session_state.algo,
                    is_classification=st.session_state.get('is_classification', False),
                    preprocessors=st.session_state.preprocessors
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
                input_values = {}
                for feature in st.session_state.feature_columns:
                    # Check if feature is categorical
                    is_categorical = feature in st.session_state.data_types.get('categorical', [])
                    
                    if is_categorical:
                        # For categorical features, show a dropdown
                        unique_values = st.session_state.data[feature].unique().tolist()
                        input_val = st.selectbox(f"Select value for {feature}", unique_values)
                    else:
                        # For numeric features, show a number input
                        input_val = st.number_input(
                            f"Enter value for {feature}", 
                            value=float(st.session_state.X[feature].mean())
                        )
                    
                    input_values[feature] = input_val
                
                # Create DataFrame with input values
                input_df = pd.DataFrame([input_values])
                
                # Prediction button
                if st.button("Predict"):
                    try:
                        prediction_result = make_prediction(
                            st.session_state.model,
                            input_df,
                            preprocessors=st.session_state.preprocessors,
                            algo_type=st.session_state.algo,
                            is_classification=st.session_state.get('is_classification', False)
                        )
                        
                        st.success("Prediction completed!")
                        
                        # Display prediction result
                        if isinstance(prediction_result, dict):
                            if 'class' in prediction_result:
                                st.write(f"### Predicted Class: {prediction_result['class']}")
                                
                                if 'probabilities' in prediction_result:
                                    st.write("#### Class Probabilities:")
                                    for class_label, prob in prediction_result['probabilities'].items():
                                        st.write(f"{class_label}: {prob}")
                            elif 'value' in prediction_result:
                                st.write(f"### Predicted Value: {prediction_result['value']:.4f}")
                        else:
                            st.write(f"### {prediction_result}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        if isinstance(e, MLInsightException):
                            st.error(f"Error: {e.message}")
                            if e.details:
                                st.error(f"Details: {e.details}")
                        else:
                            st.error(f"Error making prediction: {error_msg}")
                        logging.error(f"Prediction error: {error_msg}")
                        logging.error(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.write("ML Insight Lab © 2023 - Advanced Machine Learning Exploration & Visualization")

if __name__ == "__main__":
    main()