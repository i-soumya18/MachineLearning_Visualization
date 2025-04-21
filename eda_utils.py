import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import traceback

def run_eda(df):
    """
    Run comprehensive exploratory data analysis on the dataset
    
    Parameters:
    -----------
    df : DataFrame
        The dataset to analyze
    """
    try:
        st.subheader("Exploratory Data Analysis")
        
        # Basic dataset information
        with st.expander("Dataset Overview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", df.shape)
                st.write("Number of Rows:", df.shape[0])
                st.write("Number of Columns:", df.shape[1])
            with col2:
                st.write("Data Types:")
                dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                dtypes_df = dtypes_df.reset_index().rename(columns={'index': 'Column'})
                st.dataframe(dtypes_df)
        
        # Missing values analysis
        with st.expander("Missing Values Analysis", expanded=True):
            missing_values = df.isna().sum()
            missing_percent = (missing_values / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage': missing_percent
            })
            missing_df = missing_df.reset_index().rename(columns={'index': 'Column'})
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if not missing_df.empty:
                st.write("Columns with Missing Values:")
                st.dataframe(missing_df)
                
                # Visualize missing values
                fig = px.bar(missing_df, x='Column', y='Percentage', 
                           title='Percentage of Missing Values by Column',
                           labels={'Percentage': 'Missing Values (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
        
        # Numerical columns analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            with st.expander("Numerical Features Analysis", expanded=True):
                # Statistical summary
                st.write("Statistical Summary:")
                stats_df = df[numeric_cols].describe().T
                st.dataframe(stats_df)
                
                # Distribution plots
                st.write("Distribution of Numerical Features:")
                
                # Allow user to select columns for visualization
                selected_num_cols = st.multiselect(
                    "Select numerical columns to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_num_cols:
                    # Create distribution plots
                    for col in selected_num_cols:
                        fig = make_subplots(rows=1, cols=2, 
                                          subplot_titles=["Histogram", "Box Plot"],
                                          specs=[[{"type": "xy"}, {"type": "xy"}]])
                        
                        # Histogram
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col, nbinsx=30),
                            row=1, col=1
                        )
                        
                        # Box plot
                        fig.add_trace(
                            go.Box(y=df[col], name=col),
                            row=1, col=2
                        )
                        
                        fig.update_layout(
                            title=f"Distribution of {col}",
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation heatmap
                    if len(selected_num_cols) > 1:
                        st.write("Correlation Matrix:")
                        corr_matrix = df[selected_num_cols].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            title="Correlation Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter plot matrix for selected columns (limit to 5 to avoid overload)
                        if len(selected_num_cols) <= 5:
                            st.write("Scatter Plot Matrix:")
                            fig = px.scatter_matrix(
                                df[selected_num_cols],
                                dimensions=selected_num_cols,
                                title="Scatter Plot Matrix"
                            )
                            fig.update_layout(height=800)
                            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if categorical_cols:
            with st.expander("Categorical Features Analysis", expanded=True):
                # Allow user to select columns for visualization
                selected_cat_cols = st.multiselect(
                    "Select categorical columns to visualize",
                    categorical_cols,
                    default=categorical_cols[:min(5, len(categorical_cols))]
                )
                
                if selected_cat_cols:
                    for col in selected_cat_cols:
                        # Value counts
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = [col, 'Count']
                        
                        st.write(f"Value Counts for {col}:")
                        st.dataframe(value_counts)
                        
                        # Bar chart
                        fig = px.bar(
                            value_counts,
                            x=col,
                            y='Count',
                            title=f"Distribution of {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pie chart
                        fig = px.pie(
                            value_counts,
                            values='Count',
                            names=col,
                            title=f"Proportion of {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Bivariate analysis
        if numeric_cols and len(numeric_cols) >= 2:
            with st.expander("Bivariate Analysis", expanded=True):
                st.write("Explore relationships between variables:")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Select X variable", numeric_cols)
                with col2:
                    y_var = st.selectbox("Select Y variable", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                if x_var and y_var:
                    # Scatter plot
                    fig = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        title=f"Scatter Plot: {x_var} vs {y_var}",
                        trendline="ols"  # Add trend line
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    corr = df[x_var].corr(df[y_var])
                    st.write(f"Correlation between {x_var} and {y_var}: {corr:.4f}")
                
                # If categorical columns exist, allow for grouped analysis
                if categorical_cols:
                    color_var = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    
                    if color_var != "None":
                        # Scatter plot with color
                        fig = px.scatter(
                            df,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            title=f"Scatter Plot: {x_var} vs {y_var} (grouped by {color_var})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Box plots by group
                        fig = px.box(
                            df,
                            x=color_var,
                            y=y_var,
                            title=f"Box Plot: {y_var} by {color_var}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection
        if numeric_cols:
            with st.expander("Outlier Detection", expanded=True):
                st.write("Detect outliers in numerical features:")
                
                selected_outlier_col = st.selectbox(
                    "Select column for outlier detection",
                    numeric_cols
                )
                
                if selected_outlier_col:
                    # Z-score method
                    z_scores = np.abs(stats.zscore(df[selected_outlier_col].dropna()))
                    outliers_z = np.where(z_scores > 3)[0]
                    
                    # IQR method
                    Q1 = df[selected_outlier_col].quantile(0.25)
                    Q3 = df[selected_outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_iqr = df[(df[selected_outlier_col] < (Q1 - 1.5 * IQR)) | 
                                     (df[selected_outlier_col] > (Q3 + 1.5 * IQR))].index
                    
                    st.write(f"Number of outliers detected (Z-score method, |z| > 3): {len(outliers_z)}")
                    st.write(f"Number of outliers detected (IQR method): {len(outliers_iqr)}")
                    
                    # Box plot with outliers highlighted
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=df[selected_outlier_col],
                        name=selected_outlier_col,
                        boxpoints='outliers',  # Show only outliers
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(
                            color='red',
                            size=5
                        )
                    ))
                    fig.update_layout(
                        title=f"Box Plot with Outliers: {selected_outlier_col}",
                        yaxis_title=selected_outlier_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram with outlier thresholds
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[selected_outlier_col],
                        name=selected_outlier_col,
                        nbinsx=30
                    ))
                    
                    # Add IQR threshold lines
                    fig.add_vline(x=Q1 - 1.5 * IQR, line_dash="dash", line_color="red",
                                annotation_text="Lower IQR Threshold")
                    fig.add_vline(x=Q3 + 1.5 * IQR, line_dash="dash", line_color="red",
                                annotation_text="Upper IQR Threshold")
                    
                    fig.update_layout(
                        title=f"Histogram with Outlier Thresholds: {selected_outlier_col}",
                        xaxis_title=selected_outlier_col,
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis if date columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            with st.expander("Time Series Analysis", expanded=True):
                st.write("Analyze time series data:")
                
                date_col = st.selectbox("Select date column", date_cols)
                value_col = st.selectbox("Select value column for time series", numeric_cols)
                
                if date_col and value_col:
                    # Ensure data is sorted by date
                    df_ts = df.sort_values(by=date_col)
                    
                    # Time series plot
                    fig = px.line(
                        df_ts,
                        x=date_col,
                        y=value_col,
                        title=f"Time Series: {value_col} over {date_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Resample by different time periods
                    st.write("Resample time series:")
                    resample_period = st.selectbox(
                        "Select resampling period",
                        ["Day", "Week", "Month", "Quarter", "Year"]
                    )
                    
                    resample_map = {
                        "Day": 'D',
                        "Week": 'W',
                        "Month": 'M',
                        "Quarter": 'Q',
                        "Year": 'Y'
                    }
                    
                    # Set date column as index for resampling
                    df_ts_indexed = df_ts.set_index(date_col)
                    
                    # Resample and calculate mean
                    df_resampled = df_ts_indexed[value_col].resample(resample_map[resample_period]).mean()
                    df_resampled = df_resampled.reset_index()
                    
                    # Plot resampled data
                    fig = px.line(
                        df_resampled,
                        x=date_col,
                        y=value_col,
                        title=f"{value_col} Resampled by {resample_period}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature relationships
        with st.expander("Feature Relationships", expanded=True):
            st.write("Explore relationships between multiple features:")
            
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_var = st.selectbox("Select X axis", numeric_cols, key="rel_x")
                with col2:
                    y_var = st.selectbox("Select Y axis", numeric_cols, index=1, key="rel_y")
                with col3:
                    z_var = st.selectbox("Select Z axis (size)", numeric_cols, index=2, key="rel_z")
                
                if categorical_cols:
                    color_var = st.selectbox("Select color variable (optional)", ["None"] + categorical_cols, key="rel_color")
                    
                    if color_var != "None":
                        fig = px.scatter(
                            df,
                            x=x_var,
                            y=y_var,
                            size=z_var,
                            color=color_var,
                            title=f"Bubble Chart: {x_var} vs {y_var} (size: {z_var}, color: {color_var})"
                        )
                    else:
                        fig = px.scatter(
                            df,
                            x=x_var,
                            y=y_var,
                            size=z_var,
                            title=f"Bubble Chart: {x_var} vs {y_var} (size: {z_var})"
                        )
                else:
                    fig = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        size=z_var,
                        title=f"Bubble Chart: {x_var} vs {y_var} (size: {z_var})"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D scatter plot if enough numeric columns
            if len(numeric_cols) >= 3:
                st.write("3D Visualization:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_var_3d = st.selectbox("Select X axis (3D)", numeric_cols, key="3d_x")
                with col2:
                    y_var_3d = st.selectbox("Select Y axis (3D)", numeric_cols, index=1, key="3d_y")
                with col3:
                    z_var_3d = st.selectbox("Select Z axis (3D)", numeric_cols, index=2, key="3d_z")
                
                if categorical_cols:
                    color_var_3d = st.selectbox("Select color variable (3D)", ["None"] + categorical_cols, key="3d_color")
                    
                    if color_var_3d != "None":
                        fig = px.scatter_3d(
                            df,
                            x=x_var_3d,
                            y=y_var_3d,
                            z=z_var_3d,
                            color=color_var_3d,
                            title=f"3D Scatter Plot: {x_var_3d} vs {y_var_3d} vs {z_var_3d}"
                        )
                    else:
                        fig = px.scatter_3d(
                            df,
                            x=x_var_3d,
                            y=y_var_3d,
                            z=z_var_3d,
                            title=f"3D Scatter Plot: {x_var_3d} vs {y_var_3d} vs {z_var_3d}"
                        )
                else:
                    fig = px.scatter_3d(
                        df,
                        x=x_var_3d,
                        y=y_var_3d,
                        z=z_var_3d,
                        title=f"3D Scatter Plot: {x_var_3d} vs {y_var_3d} vs {z_var_3d}"
                    )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during EDA: {str(e)}")
        logging.error(f"EDA error: {str(e)}")
        logging.error(traceback.format_exc())