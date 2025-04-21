import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import logging
import traceback

class ProgressCallback:
    """
    Callback class to track and display training progress
    """
    def __init__(self, total_iterations, progress_bar=None, metric_container=None):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.progress_bar = progress_bar
        self.metric_container = metric_container
        self.metrics_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'r2': [],
            'val_r2': []
        }
        self.start_time = time.time()
    
    def update(self, iteration=None, metrics=None):
        """Update progress bar and metrics"""
        if iteration is not None:
            self.current_iteration = iteration
        else:
            self.current_iteration += 1
        
        # Update progress bar
        if self.progress_bar:
            progress = min(self.current_iteration / self.total_iterations, 1.0)
            self.progress_bar.progress(progress)
            
            # Calculate ETA
            elapsed_time = time.time() - self.start_time
            if progress > 0:
                eta = elapsed_time / progress * (1 - progress)
                eta_text = f"ETA: {eta:.1f}s" if eta < 60 else f"ETA: {eta/60:.1f}m"
            else:
                eta_text = "ETA: calculating..."
            
            # Display iteration info
            self.progress_bar.text(f"Iteration {self.current_iteration}/{self.total_iterations} ({progress*100:.1f}%) - {eta_text}")
        
        # Update metrics
        if metrics and self.metric_container:
            # Store metrics history
            for key, value in metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            # Display current metrics
            metric_text = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if v is not None])
            self.metric_container.text(f"Metrics: {metric_text}")

def train_with_progress(model, X, y, is_classification=False, test_size=0.2, epochs=10, batch_size=None):
    """
    Train a model with progress tracking
    
    Parameters:
    -----------
    model : estimator
        The model to train
    X : array-like
        The feature dataset
    y : array-like
        The target values
    is_classification : bool
        Whether this is a classification problem
    test_size : float
        Proportion of data to use for validation
    epochs : int
        Number of training epochs (iterations)
    batch_size : int or None
        Batch size for training. If None, use all data.
    
    Returns:
    --------
    tuple
        (trained_model, metrics, history)
    """
    try:
        # Create progress containers
        progress_container = st.empty()
        metrics_container = st.empty()
        chart_container = st.empty()
        
        with progress_container.container():
            st.write("Training Progress:")
            progress_bar = st.progress(0.0)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Initialize callback
        callback = ProgressCallback(
            total_iterations=epochs,
            progress_bar=progress_bar,
            metric_container=metrics_container
        )
        
        # Clone the model to avoid modifying the original
        trained_model = clone(model)
        
        # Training loop
        for epoch in range(epochs):
            # If batch_size is provided, train in batches
            if batch_size:
                # Shuffle indices
                indices = np.random.permutation(len(X_train))
                
                # Train in batches
                for start_idx in range(0, len(indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    X_batch = X_train.iloc[batch_indices] if hasattr(X_train, 'iloc') else X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Partial fit if available, otherwise fit
                    if hasattr(trained_model, 'partial_fit'):
                        if is_classification:
                            # For classification, we need to provide all classes
                            classes = np.unique(y)
                            trained_model.partial_fit(X_batch, y_batch, classes=classes)
                        else:
                            trained_model.partial_fit(X_batch, y_batch)
                    else:
                        # If partial_fit is not available, just fit on the batch
                        # Note: This will overwrite previous training, not ideal for all models
                        trained_model.fit(X_batch, y_batch)
            else:
                # Train on all data at once
                trained_model.fit(X_train, y_train)
            
            # Calculate metrics
            y_train_pred = trained_model.predict(X_train)
            y_val_pred = trained_model.predict(X_val)
            
            metrics = {}
            
            if is_classification:
                # Classification metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                val_acc = accuracy_score(y_val, y_val_pred)
                
                metrics['accuracy'] = train_acc
                metrics['val_accuracy'] = val_acc
                
                # Add loss if model has predict_proba
                if hasattr(trained_model, 'predict_proba'):
                    try:
                        y_train_proba = trained_model.predict_proba(X_train)
                        y_val_proba = trained_model.predict_proba(X_val)
                        
                        # Calculate log loss (cross-entropy)
                        from sklearn.metrics import log_loss
                        train_loss = log_loss(y_train, y_train_proba)
                        val_loss = log_loss(y_val, y_val_proba)
                        
                        metrics['loss'] = train_loss
                        metrics['val_loss'] = val_loss
                    except Exception as e:
                        logging.warning(f"Could not calculate loss: {str(e)}")
            else:
                # Regression metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                val_mse = mean_squared_error(y_val, y_val_pred)
                
                train_r2 = r2_score(y_train, y_train_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                
                metrics['loss'] = train_mse
                metrics['val_loss'] = val_mse
                metrics['r2'] = train_r2
                metrics['val_r2'] = val_r2
            
            # Update progress
            callback.update(epoch + 1, metrics)
            
            # Update chart
            with chart_container:
                # Create metrics chart
                if epoch > 0:  # Only show chart after first epoch
                    history_df = pd.DataFrame({
                        'epoch': list(range(1, epoch + 2)),
                        **{k: v for k, v in callback.metrics_history.items() if len(v) > 0}
                    })
                    
                    # Determine which metrics to plot
                    if is_classification and 'accuracy' in history_df.columns:
                        import plotly.express as px
                        
                        fig = px.line(
                            history_df, 
                            x='epoch', 
                            y=['accuracy', 'val_accuracy'],
                            title='Training and Validation Accuracy',
                            labels={'value': 'Accuracy', 'variable': 'Metric'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if 'loss' in history_df.columns:
                            fig = px.line(
                                history_df, 
                                x='epoch', 
                                y=['loss', 'val_loss'],
                                title='Training and Validation Loss',
                                labels={'value': 'Loss', 'variable': 'Metric'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif 'r2' in history_df.columns:
                        import plotly.express as px
                        
                        fig = px.line(
                            history_df, 
                            x='epoch', 
                            y=['r2', 'val_r2'],
                            title='Training and Validation R²',
                            labels={'value': 'R²', 'variable': 'Metric'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig = px.line(
                            history_df, 
                            x='epoch', 
                            y=['loss', 'val_loss'],
                            title='Training and Validation MSE',
                            labels={'value': 'MSE', 'variable': 'Metric'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Small delay to allow UI to update
            time.sleep(0.1)
        
        # Final metrics
        final_metrics = {}
        
        if is_classification:
            y_pred = trained_model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            final_metrics['accuracy'] = accuracy
            
            # Add more classification metrics
            from sklearn.metrics import classification_report, confusion_matrix
            final_metrics['classification_report'] = classification_report(y, y_pred, output_dict=True)
            final_metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        else:
            y_pred = trained_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            final_metrics['mse'] = mse
            final_metrics['r2'] = r2
        
        # Return the trained model, metrics, and history
        return trained_model, final_metrics, callback.metrics_history
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        logging.error(f"Training error: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None