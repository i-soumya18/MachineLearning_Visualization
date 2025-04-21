import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import logging
import traceback

def create_advanced_visualizations(X, y, model, algo, is_classification=False, preprocessors=None, metrics=None):
    """
    Create advanced visualizations for different algorithm types
    
    Parameters:
    -----------
    X : DataFrame
        The feature dataset
    y : array-like
        The target values
    model : object
        The trained model
    algo : str
        Algorithm name
    is_classification : bool
        Whether this is a classification problem
    preprocessors : dict
        Preprocessing objects
    metrics : dict
        Model performance metrics
    """
    try:
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Model Performance", "Feature Importance", "Model Specific", "Prediction Analysis"])
        
        with viz_tabs[0]:  # Model Performance
            st.subheader("Model Performance Visualization")
            
            if is_classification:
                # Classification metrics visualization
                if metrics and 'confusion_matrix' in metrics:
                    cm = np.array(metrics['confusion_matrix'])
                    
                    # Create a heatmap with annotations
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add normalized confusion matrix
                    if cm.sum() > 0:
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        fig = px.imshow(
                            cm_norm,
                            text_auto='.2f',
                            color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="Actual", color="Proportion"),
                            title="Normalized Confusion Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ROC curve if probabilities are available
                if hasattr(model, 'predict_proba'):
                    st.write("ROC Curve:")
                    
                    try:
                        # Get class probabilities
                        y_prob = model.predict_proba(X)
                        
                        # For binary classification
                        if y_prob.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC Curve (AUC = {roc_auc:.3f})'
                            ))
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                line=dict(dash='dash', color='gray'),
                                name='Random'
                            ))
                            fig.update_layout(
                                title='Receiver Operating Characteristic (ROC) Curve',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                yaxis=dict(scaleanchor="x", scaleratio=1),
                                xaxis=dict(constrain='domain')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Precision-Recall curve
                            precision, recall, _ = precision_recall_curve(y, y_prob[:, 1])
                            pr_auc = auc(recall, precision)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=recall, y=precision,
                                mode='lines',
                                name=f'PR Curve (AUC = {pr_auc:.3f})'
                            ))
                            fig.update_layout(
                                title='Precision-Recall Curve',
                                xaxis_title='Recall',
                                yaxis_title='Precision'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # For multiclass classification
                        else:
                            st.write("Multiclass ROC curves are not displayed due to complexity.")
                    except Exception as e:
                        st.warning(f"Could not generate ROC curve: {str(e)}")
            
            else:
                # Regression metrics visualization
                if metrics and 'mse' in metrics:
                    # Actual vs Predicted plot
                    y_pred = model.predict(X)
                    
                    fig = px.scatter(
                        x=y, y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted Values'
                    )
                    
                    # Add perfect prediction line
                    min_val = min(min(y), min(y_pred))
                    max_val = max(max(y), max(y_pred))
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual plot
                    residuals = y - y_pred
                    
                    fig = px.scatter(
                        x=y_pred, y=residuals,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title='Residual Plot'
                    )
                    
                    # Add horizontal line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual distribution
                    fig = px.histogram(
                        residuals,
                        nbins=30,
                        title='Residual Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Q-Q plot for residuals
                    qq_x = np.linspace(np.min(residuals), np.max(residuals), 100)
                    qq_y = np.quantile(np.random.normal(0, np.std(residuals), len(residuals)), 
                                      np.linspace(0, 1, 100))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=qq_x, y=qq_y,
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Normal Distribution'
                    ))
                    
                    # Add actual quantiles
                    sorted_residuals = np.sort(residuals)
                    p = np.linspace(0, 1, len(sorted_residuals))
                    theoretical_quantiles = np.quantile(np.random.normal(0, np.std(residuals), len(residuals)), p)
                    
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles, y=sorted_residuals,
                        mode='markers',
                        name='Residuals'
                    ))
                    
                    fig.update_layout(
                        title='Q-Q Plot of Residuals',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:  # Feature Importance
            st.subheader("Feature Importance Analysis")
            
            # Check if model has feature_importances_ attribute (tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                feature_names = X.columns
                
                # Create DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Bar chart of feature importance
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # For linear models, use coefficients
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                
                # Handle different shapes of coefficients
                if coef.ndim > 1 and coef.shape[0] > 1:
                    st.write("This model has multiple coefficient sets (e.g., multiclass). Showing first set.")
                    coef = coef[0]
                
                feature_names = X.columns
                
                # Create DataFrame for visualization
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coef
                })
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                # Bar chart of coefficients
                fig = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Model Coefficients'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add intercept if available
                if hasattr(model, 'intercept_'):
                    st.write(f"Intercept: {model.intercept_}")
            
            # For models without direct feature importance, use permutation importance
            else:
                st.write("Computing permutation feature importance...")
                
                try:
                    # Calculate permutation importance
                    perm_importance = permutation_importance(
                        model, X, y, n_repeats=10, random_state=42
                    )
                    
                    # Create DataFrame for visualization
                    perm_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': perm_importance.importances_mean
                    })
                    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
                    
                    # Bar chart of permutation importance
                    fig = px.bar(
                        perm_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Permutation Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute permutation importance: {str(e)}")
        
        with viz_tabs[2]:  # Model Specific
            st.subheader("Algorithm-Specific Visualizations")
            
            # Different visualizations based on algorithm type
            if algo in ['linear', 'linear_regression']:
                if X.shape[1] == 1:
                    # Simple linear regression visualization
                    feature = X.columns[0]
                    
                    # Create scatter plot with regression line
                    fig = px.scatter(
                        x=X[feature], y=y,
                        labels={'x': feature, 'y': 'Target'},
                        title='Linear Regression'
                    )
                    
                    # Add regression line
                    x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
                    x_range_df = pd.DataFrame({feature: x_range})
                    
                    # Fill in other features with mean values if needed
                    for col in X.columns:
                        if col != feature:
                            x_range_df[col] = X[col].mean()
                    
                    y_pred = model.predict(x_range_df)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name='Regression Line',
                        line=dict(color='red')
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show equation
                    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                        intercept = model.intercept_
                        equation = f"y = {coef:.4f} * {feature} + {intercept:.4f}"
                        st.write("Regression Equation:", equation)
                
                elif X.shape[1] == 2:
                    # 3D visualization for 2 features
                    features = X.columns.tolist()
                    
                    # Create 3D scatter plot with regression plane
                    fig = go.Figure()
                    
                    # Add scatter points
                    fig.add_trace(go.Scatter3d(
                        x=X[features[0]],
                        y=X[features[1]],
                        z=y,
                        mode='markers',
                        marker=dict(size=5),
                        name='Data Points'
                    ))
                    
                    # Create mesh grid for regression plane
                    x_range = np.linspace(X[features[0]].min(), X[features[0]].max(), 20)
                    y_range = np.linspace(X[features[1]].min(), X[features[1]].max(), 20)
                    xx, yy = np.meshgrid(x_range, y_range)
                    
                    # Create input data for prediction
                    grid_df = pd.DataFrame({
                        features[0]: xx.ravel(),
                        features[1]: yy.ravel()
                    })
                    
                    # Fill in other features with mean values if needed
                    for col in X.columns:
                        if col not in features:
                            grid_df[col] = X[col].mean()
                    
                    # Predict z values
                    zz = model.predict(grid_df).reshape(xx.shape)
                    
                    # Add regression plane
                    fig.add_trace(go.Surface(
                        x=xx, y=yy, z=zz,
                        opacity=0.7,
                        colorscale='Viridis',
                        name='Regression Plane'
                    ))
                    
                    fig.update_layout(
                        title='Linear Regression 3D Visualization',
                        scene=dict(
                            xaxis_title=features[0],
                            yaxis_title=features[1],
                            zaxis_title='Target'
                        ),
                        height=700
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.write("Visualization limited for high-dimensional data.")
                    
                    # Partial dependence plots
                    st.write("Select a feature for partial dependence plot:")
                    selected_feature = st.selectbox("Feature", X.columns)
                    
                    if selected_feature:
                        # Create range of values for selected feature
                        feature_range = np.linspace(
                            X[selected_feature].min(),
                            X[selected_feature].max(),
                            50
                        )
                        
                        # Create prediction data
                        pred_data = []
                        for val in feature_range:
                            # Create a copy of the dataset with the selected feature varied
                            X_copy = X.copy()
                            X_copy[selected_feature] = val
                            
                            # Predict and take mean
                            pred = model.predict(X_copy).mean()
                            pred_data.append(pred)
                        
                        # Create partial dependence plot
                        fig = px.line(
                            x=feature_range,
                            y=pred_data,
                            labels={'x': selected_feature, 'y': 'Predicted Target (avg)'},
                            title=f'Partial Dependence Plot for {selected_feature}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['logistic', 'logistic_regression']:
                if X.shape[1] <= 2:
                    # Decision boundary visualization for logistic regression
                    if X.shape[1] == 1:
                        # 1D feature case
                        feature = X.columns[0]
                        
                        # Create range of values
                        x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
                        x_range_df = pd.DataFrame({feature: x_range})
                        
                        # Predict probabilities
                        y_probs = model.predict_proba(x_range_df)[:, 1]
                        
                        # Plot probability curve
                        fig = go.Figure()
                        
                        # Add scatter points (jittered for visibility)
                        jitter = 0.05 * (max(y) - min(y))
                        fig.add_trace(go.Scatter(
                            x=X[feature],
                            y=y + np.random.uniform(-jitter, jitter, size=len(y)),
                            mode='markers',
                            name='Data Points',
                            marker=dict(
                                color=y,
                                colorscale='Viridis',
                                size=8
                            )
                        ))
                        
                        # Add probability curve
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_probs,
                            mode='lines',
                            name='Probability',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add decision boundary at p=0.5
                        fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                                    annotation_text="Decision Boundary (p=0.5)")
                        
                        fig.update_layout(
                            title='Logistic Regression Probability Curve',
                            xaxis_title=feature,
                            yaxis_title='Probability / Class'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif X.shape[1] == 2:
                        # 2D feature case - decision boundary
                        features = X.columns.tolist()
                        
                        # Create mesh grid for decision boundary
                        x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
                        y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                           np.linspace(y_min, y_max, 100))
                        
                        # Create input data for prediction
                        grid_df = pd.DataFrame({
                            features[0]: xx.ravel(),
                            features[1]: yy.ravel()
                        })
                        
                        # Predict probabilities
                        Z = model.predict_proba(grid_df)[:, 1].reshape(xx.shape)
                        
                        # Create contour plot
                        fig = go.Figure()
                        
                        # Add contour for probability
                        fig.add_trace(go.Contour(
                            z=Z,
                            x=np.linspace(x_min, x_max, 100),
                            y=np.linspace(y_min, y_max, 100),
                            colorscale='RdBu_r',
                            opacity=0.8,
                            showscale=True,
                            contours=dict(
                                start=0,
                                end=1,
                                size=0.05
                            ),
                            colorbar=dict(
                                title='Probability',
                                titleside='right'
                            )
                        ))
                        
                        # Add decision boundary at p=0.5
                        fig.add_trace(go.Contour(
                            z=Z,
                            x=np.linspace(x_min, x_max, 100),
                            y=np.linspace(y_min, y_max, 100),
                            showscale=False,
                            contours=dict(
                                start=0.5,
                                end=0.5,
                                size=0,
                                coloring='lines',
                                showlabels=True
                            ),
                            line=dict(
                                color='black',
                                width=2
                            ),
                            name='Decision Boundary'
                        ))
                        
                        # Add scatter points
                        fig.add_trace(go.Scatter(
                            x=X[features[0]],
                            y=X[features[1]],
                            mode='markers',
                            marker=dict(
                                color=y,
                                colorscale='Viridis',
                                size=10,
                                line=dict(color='black', width=1)
                            ),
                            name='Data Points'
                        ))
                        
                        fig.update_layout(
                            title='Logistic Regression Decision Boundary',
                            xaxis_title=features[0],
                            yaxis_title=features[1],
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.write("Visualization limited for high-dimensional data.")
                    
                    # Probability distribution
                    y_probs = model.predict_proba(X)
                    
                    # Create histogram of probabilities
                    fig = go.Figure()
                    
                    for i in range(y_probs.shape[1]):
                        fig.add_trace(go.Histogram(
                            x=y_probs[:, i],
                            name=f'Class {i}',
                            opacity=0.7,
                            nbinsx=30
                        ))
                    
                    fig.update_layout(
                        title='Probability Distribution by Class',
                        xaxis_title='Probability',
                        yaxis_title='Count',
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['dtree', 'decision_tree']:
                st.write("Decision Tree Visualization")
                
                # For small trees, we can visualize the structure
                if hasattr(model, 'tree_'):
                    from sklearn.tree import plot_tree
                    import matplotlib.pyplot as plt
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot tree
                    plot_tree(
                        model,
                        feature_names=X.columns,
                        filled=True,
                        rounded=True,
                        ax=ax,
                        max_depth=3  # Limit depth for readability
                    )
                    
                    st.pyplot(fig)
                    
                    # Add note about depth limitation
                    st.info("Note: Tree visualization is limited to depth 3 for readability.")
                
                # Feature importance for decision trees
                if hasattr(model, 'feature_importances_'):
                    st.write("Decision Tree Feature Importance")
                    
                    # Create DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Bar chart of feature importance
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Decision Tree Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['rf', 'random_forest']:
                st.write("Random Forest Visualization")
                
                # Feature importance for random forest
                if hasattr(model, 'feature_importances_'):
                    st.write("Random Forest Feature Importance")
                    
                    # Create DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Bar chart of feature importance
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Random Forest Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tree depth analysis
                if hasattr(model, 'estimators_'):
                    depths = [estimator.get_depth() for estimator in model.estimators_]
                    
                    # Histogram of tree depths
                    fig = px.histogram(
                        depths,
                        nbins=20,
                        title='Distribution of Tree Depths in Random Forest'
                    )
                    fig.update_layout(
                        xaxis_title='Tree Depth',
                        yaxis_title='Count'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['svm']:
                if X.shape[1] <= 2:
                    # SVM decision boundary visualization
                    if X.shape[1] == 2:
                        features = X.columns.tolist()
                        
                        # Create mesh grid for decision boundary
                        x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
                        y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                           np.linspace(y_min, y_max, 100))
                        
                        # Create input data for prediction
                        grid_df = pd.DataFrame({
                            features[0]: xx.ravel(),
                            features[1]: yy.ravel()
                        })
                        
                        # Predict class
                        Z = model.predict(grid_df).reshape(xx.shape)
                        
                        # Create contour plot
                        fig = go.Figure()
                        
                        # Add contour for decision regions
                        fig.add_trace(go.Contour(
                            z=Z,
                            x=np.linspace(x_min, x_max, 100),
                            y=np.linspace(y_min, y_max, 100),
                            colorscale='Viridis',
                            showscale=False,
                            contours=dict(
                                coloring='heatmap'
                            )
                        ))
                        
                        # Add scatter points
                        fig.add_trace(go.Scatter(
                            x=X[features[0]],
                            y=X[features[1]],
                            mode='markers',
                            marker=dict(
                                color=y,
                                colorscale='Viridis',
                                size=10,
                                line=dict(color='black', width=1)
                            ),
                            name='Data Points'
                        ))
                        
                        fig.update_layout(
                            title='SVM Decision Boundary',
                            xaxis_title=features[0],
                            yaxis_title=features[1],
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.write("Visualization limited for high-dimensional SVM.")
                    
                    # If SVM has decision function, visualize it
                    if hasattr(model, 'decision_function'):
                        decision_values = model.decision_function(X)
                        
                        # For binary classification
                        if decision_values.ndim == 1:
                            fig = px.histogram(
                                decision_values,
                                color=y,
                                nbins=50,
                                title='SVM Decision Function Distribution',
                                labels={'value': 'Decision Value', 'color': 'Class'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['kmeans']:
                st.write("K-Means Clustering Visualization")
                
                if X.shape[1] >= 2:
                    # Select features for visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feature = st.selectbox("X-axis feature", X.columns, key="kmeans_x")
                    with col2:
                        y_feature = st.selectbox("Y-axis feature", X.columns, index=1, key="kmeans_y")
                    
                    # Get cluster labels and centers
                    labels = model.labels_
                    centers = model.cluster_centers_
                    
                    # Create scatter plot with clusters
                    fig = px.scatter(
                        x=X[x_feature],
                        y=X[y_feature],
                        color=[str(label) for label in labels],
                        title='K-Means Clustering',
                        labels={'x': x_feature, 'y': y_feature, 'color': 'Cluster'}
                    )
                    
                    # Add cluster centers
                    for i, center in enumerate(centers):
                        # Find index of x_feature and y_feature in the original feature list
                        x_idx = list(X.columns).index(x_feature)
                        y_idx = list(X.columns).index(y_feature)
                        
                        fig.add_trace(go.Scatter(
                            x=[center[x_idx]],
                            y=[center[y_idx]],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=15,
                                color='black',
                                line=dict(width=2)
                            ),
                            name=f'Cluster {i} Center'
                        ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D visualization if there are at least 3 features
                    if X.shape[1] >= 3:
                        st.write("3D Cluster Visualization")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_feature_3d = st.selectbox("X-axis feature (3D)", X.columns, key="kmeans_3d_x")
                        with col2:
                            y_feature_3d = st.selectbox("Y-axis feature (3D)", X.columns, index=1, key="kmeans_3d_y")
                        with col3:
                            z_feature_3d = st.selectbox("Z-axis feature (3D)", X.columns, index=2, key="kmeans_3d_z")
                        
                        # Create 3D scatter plot
                        fig = px.scatter_3d(
                            x=X[x_feature_3d],
                            y=X[y_feature_3d],
                            z=X[z_feature_3d],
                            color=[str(label) for label in labels],
                            title='K-Means Clustering (3D)',
                            labels={'x': x_feature_3d, 'y': y_feature_3d, 'z': z_feature_3d, 'color': 'Cluster'}
                        )
                        
                        # Add cluster centers
                        for i, center in enumerate(centers):
                            # Find indices of features
                            x_idx = list(X.columns).index(x_feature_3d)
                            y_idx = list(X.columns).index(y_feature_3d)
                            z_idx = list(X.columns).index(z_feature_3d)
                            
                            fig.add_trace(go.Scatter3d(
                                x=[center[x_idx]],
                                y=[center[y_idx]],
                                z=[center[z_idx]],
                                mode='markers',
                                marker=dict(
                                    symbol='x',
                                    size=8,
                                    color='black'
                                ),
                                name=f'Cluster {i} Center'
                            ))
                        
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster analysis
                    st.write("Cluster Analysis")
                    
                    # Count samples in each cluster
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    
                    # Create bar chart
                    fig = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        labels={'x': 'Cluster', 'y': 'Number of Samples'},
                        title='Samples per Cluster'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature means by cluster
                    cluster_means = pd.DataFrame()
                    for i in range(len(centers)):
                        cluster_data = X[labels == i]
                        cluster_means[f'Cluster {i}'] = cluster_data.mean()
                    
                    # Transpose for better visualization
                    cluster_means = cluster_means.T
                    
                    # Heatmap of cluster means
                    fig = px.imshow(
                        cluster_means,
                        text_auto='.2f',
                        title='Feature Means by Cluster',
                        labels=dict(x='Feature', y='Cluster', color='Mean Value')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif algo in ['dbscan']:
                st.write("DBSCAN Clustering Visualization")
                
                if X.shape[1] >= 2:
                    # Select features for visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feature = st.selectbox("X-axis feature", X.columns, key="dbscan_x")
                    with col2:
                        y_feature = st.selectbox("Y-axis feature", X.columns, index=1, key="dbscan_y")
                    
                    # Get cluster labels
                    labels = model.labels_
                    
                    # Create scatter plot with clusters
                    fig = px.scatter(
                        x=X[x_feature],
                        y=X[y_feature],
                        color=[str(label) for label in labels],
                        title='DBSCAN Clustering',
                        labels={'x': x_feature, 'y': y_feature, 'color': 'Cluster'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D visualization if there are at least 3 features
                    if X.shape[1] >= 3:
                        st.write("3D Cluster Visualization")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_feature_3d = st.selectbox("X-axis feature (3D)", X.columns, key="dbscan_3d_x")
                        with col2:
                            y_feature_3d = st.selectbox("Y-axis feature (3D)", X.columns, index=1, key="dbscan_3d_y")
                        with col3:
                            z_feature_3d = st.selectbox("Z-axis feature (3D)", X.columns, index=2, key="dbscan_3d_z")
                        
                        # Create 3D scatter plot
                        fig = px.scatter_3d(
                            x=X[x_feature_3d],
                            y=X[y_feature_3d],
                            z=X[z_feature_3d],
                            color=[str(label) for label in labels],
                            title='DBSCAN Clustering (3D)',
                            labels={'x': x_feature_3d, 'y': y_feature_3d, 'z': z_feature_3d, 'color': 'Cluster'}
                        )
                        
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster analysis
                    st.write("Cluster Analysis")
                    
                    # Count samples in each cluster
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    
                    # Create bar chart
                    fig = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        labels={'x': 'Cluster', 'y': 'Number of Samples'},
                        title='Samples per Cluster'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Note about noise points
                    noise_count = (labels == -1).sum()
                    if noise_count > 0:
                        st.info(f"DBSCAN identified {noise_count} noise points (labeled as -1).")
            
            elif algo in ['pca']:
                st.write("PCA Visualization")
                
                if hasattr(model, 'explained_variance_ratio_'):
                    # Explained variance ratio
                    explained_variance = model.explained_variance_ratio_
                    
                    # Cumulative explained variance
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    # Create DataFrame for visualization
                    variance_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                        'Explained Variance': explained_variance,
                        'Cumulative Variance': cumulative_variance
                    })
                    
                    # Bar chart of explained variance
                    fig = px.bar(
                        variance_df,
                        x='Component',
                        y='Explained Variance',
                        title='Explained Variance by Principal Component'
                    )
                    
                    # Add cumulative variance line
                    fig.add_trace(go.Scatter(
                        x=variance_df['Component'],
                        y=variance_df['Cumulative Variance'],
                        mode='lines+markers',
                        name='Cumulative Variance',
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        yaxis2=dict(
                            title='Cumulative Variance',
                            overlaying='y',
                            side='right',
                            range=[0, 1]
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Transformed data visualization
                    if hasattr(model, 'transform'):
                        # Transform the data
                        transformed_data = model.transform(X)
                        
                        # Create DataFrame for visualization
                        if transformed_data.shape[1] >= 2:
                            transformed_df = pd.DataFrame({
                                'PC1': transformed_data[:, 0],
                                'PC2': transformed_data[:, 1]
                            })
                            
                            # Add target if available
                            if y is not None:
                                transformed_df['Target'] = y
                                
                                # Scatter plot of first two components
                                fig = px.scatter(
                                    transformed_df,
                                    x='PC1',
                                    y='PC2',
                                    color='Target',
                                    title='PCA: First Two Principal Components',
                                    labels={
                                        'PC1': f'PC1 ({explained_variance[0]:.2%})',
                                        'PC2': f'PC2 ({explained_variance[1]:.2%})'
                                    }
                                )
                            else:
                                # Scatter plot without color
                                fig = px.scatter(
                                    transformed_df,
                                    x='PC1',
                                    y='PC2',
                                    title='PCA: First Two Principal Components',
                                    labels={
                                        'PC1': f'PC1 ({explained_variance[0]:.2%})',
                                        'PC2': f'PC2 ({explained_variance[1]:.2%})'
                                    }
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 3D visualization if at least 3 components
                            if transformed_data.shape[1] >= 3:
                                transformed_df['PC3'] = transformed_data[:, 2]
                                
                                if y is not None:
                                    # 3D scatter plot with color
                                    fig = px.scatter_3d(
                                        transformed_df,
                                        x='PC1',
                                        y='PC2',
                                        z='PC3',
                                        color='Target',
                                        title='PCA: First Three Principal Components',
                                        labels={
                                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                                            'PC2': f'PC2 ({explained_variance[1]:.2%})',
                                            'PC3': f'PC3 ({explained_variance[2]:.2%})'
                                        }
                                    )
                                else:
                                    # 3D scatter plot without color
                                    fig = px.scatter_3d(
                                        transformed_df,
                                        x='PC1',
                                        y='PC2',
                                        z='PC3',
                                        title='PCA: First Three Principal Components',
                                        labels={
                                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                                            'PC2': f'PC2 ({explained_variance[1]:.2%})',
                                            'PC3': f'PC3 ({explained_variance[2]:.2%})'
                                        }
                                    )
                                
                                fig.update_layout(height=700)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Component loadings
                    if hasattr(model, 'components_'):
                        st.write("PCA Component Loadings")
                        
                        # Create DataFrame for loadings
                        loadings = model.components_
                        loadings_df = pd.DataFrame(
                            loadings.T,
                            index=X.columns,
                            columns=[f'PC{i+1}' for i in range(loadings.shape[0])]
                        )
                        
                        # Heatmap of loadings
                        fig = px.imshow(
                            loadings_df,
                            text_auto='.2f',
                            title='PCA Component Loadings',
                            labels=dict(x='Principal Component', y='Feature', color='Loading')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Loadings for specific component
                        st.write("Loadings for Specific Component")
                        selected_pc = st.selectbox(
                            "Select Principal Component",
                            [f'PC{i+1}' for i in range(loadings.shape[0])]
                        )
                        
                        if selected_pc:
                            # Get loadings for selected component
                            pc_loadings = loadings_df[selected_pc].sort_values(ascending=False)
                            
                            # Bar chart of loadings
                            fig = px.bar(
                                x=pc_loadings.index,
                                y=pc_loadings.values,
                                labels={'x': 'Feature', 'y': 'Loading'},
                                title=f'Feature Loadings for {selected_pc}'
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:  # Prediction Analysis
            st.subheader("Prediction Analysis")
            
            if is_classification:
                # Classification prediction analysis
                if hasattr(model, 'predict_proba'):
                    # Get probabilities
                    y_probs = model.predict_proba(X)
                    y_pred = model.predict(X)
                    
                    # Create DataFrame for analysis
                    pred_df = pd.DataFrame({
                        'Actual': y,
                        'Predicted': y_pred
                    })
                    
                    # Add probabilities for each class
                    for i in range(y_probs.shape[1]):
                        pred_df[f'Prob_Class_{i}'] = y_probs[:, i]
                    
                    # Add correct/incorrect column
                    pred_df['Correct'] = pred_df['Actual'] == pred_df['Predicted']
                    
                    # Show sample of predictions
                    st.write("Sample Predictions:")
                    st.dataframe(pred_df.head(10))
                    
                    # Prediction confidence distribution
                    st.write("Prediction Confidence Distribution:")
                    
                    # Get max probability for each prediction
                    max_probs = np.max(y_probs, axis=1)
                    
                    # Create histogram of confidence
                    fig = px.histogram(
                        max_probs,
                        nbins=20,
                        color=pred_df['Correct'],
                        barmode='overlay',
                        title='Prediction Confidence Distribution',
                        labels={'value': 'Confidence (Max Probability)', 'color': 'Correct Prediction'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion matrix with probabilities
                    st.write("Confusion Matrix with Confidence:")
                    
                    # Create pivot table of actual vs predicted with average confidence
                    pivot_df = pred_df.pivot_table(
                        index='Actual',
                        columns='Predicted',
                        values=f'Prob_Class_{0}' if y_probs.shape[1] > 0 else 'Correct',
                        aggfunc='count'
                    ).fillna(0)
                    
                    # Heatmap of confusion matrix
                    fig = px.imshow(
                        pivot_df,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title='Confusion Matrix with Counts'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Regression prediction analysis
                y_pred = model.predict(X)
                
                # Create DataFrame for analysis
                pred_df = pd.DataFrame({
                    'Actual': y,
                    'Predicted': y_pred,
                    'Error': y - y_pred,
                    'Abs_Error': np.abs(y - y_pred)
                })
                
                # Show sample of predictions
                st.write("Sample Predictions:")
                st.dataframe(pred_df.head(10))
                
                # Error distribution
                st.write("Error Distribution:")
                
                # Create histogram of errors
                fig = px.histogram(
                    pred_df['Error'],
                    nbins=30,
                    title='Error Distribution',
                    labels={'value': 'Error (Actual - Predicted)'}
                )
                
                # Add vertical line at zero
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Error by prediction value
                st.write("Error by Prediction Value:")
                
                # Create scatter plot of error vs predicted
                fig = px.scatter(
                    x=pred_df['Predicted'],
                    y=pred_df['Error'],
                    title='Error by Prediction Value',
                    labels={'x': 'Predicted Value', 'y': 'Error'}
                )
                
                # Add horizontal line at zero
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature-wise error analysis
                st.write("Feature-wise Error Analysis:")
                
                # Allow user to select a feature
                selected_feature = st.selectbox("Select feature for error analysis", X.columns)
                
                if selected_feature:
                    # Create scatter plot of error vs feature
                    fig = px.scatter(
                        x=X[selected_feature],
                        y=pred_df['Error'],
                        title=f'Error vs {selected_feature}',
                        labels={'x': selected_feature, 'y': 'Error'}
                    )
                    
                    # Add horizontal line at zero
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=X[selected_feature],
                        y=np.poly1d(np.polyfit(X[selected_feature], pred_df['Error'], 1))(X[selected_feature]),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red')
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating advanced visualizations: {str(e)}")
        logging.error(f"Visualization error: {str(e)}")
        logging.error(traceback.format_exc())