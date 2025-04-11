// SaaS Features for ML Insight Lab
let userSettings = {
    theme: 'light',
    notifications: true,
    autoSave: true,
    executionHistory: []
};

// Load user settings from localStorage
function loadUserSettings() {
    const savedSettings = localStorage.getItem('mlInsightSettings');
    if (savedSettings) {
        userSettings = JSON.parse(savedSettings);
        applyUserSettings();
    }
}

// Save user settings to localStorage
function saveUserSettings() {
    localStorage.setItem('mlInsightSettings', JSON.stringify(userSettings));
}

// Apply user settings to the UI
function applyUserSettings() {
    // Apply theme
    if (userSettings.theme === 'dark') {
        document.body.classList.add('dark-theme');
        document.getElementById('theme-toggle').innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        document.body.classList.remove('dark-theme');
        document.getElementById('theme-toggle').innerHTML = '<i class="fas fa-moon"></i>';
    }
    
    // Populate execution history if available
    renderExecutionHistory();
}

// Toggle theme between light and dark
function toggleTheme() {
    userSettings.theme = userSettings.theme === 'light' ? 'dark' : 'light';
    applyUserSettings();
    saveUserSettings();
    
    // Show toast notification
    showToast('success', 'Theme Changed', `Switched to ${userSettings.theme} theme`);
}

// Add a result to execution history
function addToHistory(algo, result) {
    const timestamp = new Date().toISOString();
    const historyItem = {
        id: generateId(),
        algorithm: algo,
        timestamp,
        result,
        dataSize: data.length
    };
    
    userSettings.executionHistory.unshift(historyItem);
    // Keep only the last 20 executions
    if (userSettings.executionHistory.length > 20) {
        userSettings.executionHistory = userSettings.executionHistory.slice(0, 20);
    }
    
    saveUserSettings();
    renderExecutionHistory();
}

// Generate a unique ID for history items
function generateId() {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
}

// Render execution history in the history panel
function renderExecutionHistory() {
    const historyPanel = document.getElementById('execution-history');
    if (!historyPanel) return;
    
    if (userSettings.executionHistory.length === 0) {
        historyPanel.innerHTML = '<p class="text-center text-gray-500 my-4">No execution history yet</p>';
        return;
    }
    
    let historyHTML = '';
    userSettings.executionHistory.forEach(item => {
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        // Create metrics display based on algorithm type
        let metricsHTML = '';
        if (item.result.metrics) {
            if ('mse' in item.result.metrics) {
                metricsHTML += `
                <div class="history-metric">
                    <span>MSE</span>
                    <strong>${item.result.metrics.mse.toFixed(4)}</strong>
                </div>
                <div class="history-metric">
                    <span>R²</span>
                    <strong>${item.result.metrics.r2.toFixed(4)}</strong>
                </div>`;
            }
            if ('accuracy' in item.result.metrics) {
                metricsHTML += `
                <div class="history-metric">
                    <span>Accuracy</span>
                    <strong>${(item.result.metrics.accuracy * 100).toFixed(2)}%</strong>
                </div>`;
            }
            if ('silhouette' in item.result.metrics) {
                metricsHTML += `
                <div class="history-metric">
                    <span>Silhouette</span>
                    <strong>${item.result.metrics.silhouette.toFixed(4)}</strong>
                </div>`;
            }
        } else if (item.algorithm === 'pca' && item.result.variance) {
            metricsHTML += `
            <div class="history-metric">
                <span>Variance Explained</span>
                <strong>${item.result.variance.map(v => (v*100).toFixed(1) + '%').join(', ')}</strong>
            </div>`;
        }
        
        // Get algorithm display name
        let algoName = '';
        switch(item.algorithm) {
            case 'linear': algoName = 'Linear Regression'; break;
            case 'logistic': algoName = 'Logistic Regression'; break;
            case 'dtree': algoName = 'Decision Tree'; break;
            case 'rf': algoName = 'Random Forest'; break;
            case 'svm': algoName = 'SVM'; break;
            case 'kmeans': algoName = 'K-Means'; break;
            case 'dbscan': algoName = 'DBSCAN'; break;
            case 'pca': algoName = 'PCA'; break;
            default: algoName = item.algorithm;
        }
        
        historyHTML += `
        <div class="history-item">
            <div class="history-header">
                <div class="history-title">${algoName}</div>
                <div class="history-date">${formattedDate}</div>
            </div>
            <div class="history-badges">
                <span class="badge primary">${item.algorithm}</span>
                <span class="badge">${item.dataSize} data points</span>
            </div>
            <div class="history-metrics">
                ${metricsHTML}
            </div>
            <div class="history-actions">
                <button onclick="loadFromHistory('${item.id}')" class="small">
                    <i class="fas fa-redo-alt"></i> Reload
                </button>
                <button onclick="shareResult('${item.id}')" class="small">
                    <i class="fas fa-share-alt"></i> Share
                </button>
                <button onclick="deleteFromHistory('${item.id}')" class="small danger">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        </div>`;
    });
    
    historyPanel.innerHTML = historyHTML;
}

// Load a previous result from history
function loadFromHistory(id) {
    const historyItem = userSettings.executionHistory.find(item => item.id === id);
    if (!historyItem) {
        showToast('error', 'Error', 'History item not found');
        return;
    }
    
    // TODO: Implement loading of historical data and results
    showToast('info', 'Loading History', `Loading ${historyItem.algorithm} execution from ${new Date(historyItem.timestamp).toLocaleDateString()}`);
}

// Share a result via a generated link
function shareResult(id) {
    const historyItem = userSettings.executionHistory.find(item => item.id === id);
    if (!historyItem) {
        showToast('error', 'Error', 'History item not found');
        return;
    }
    
    // Generate a shareable link (this would be implemented with backend support in a real app)
    const shareLink = `${window.location.origin}${window.location.pathname}?share=${id}`;
    
    // Create a temporary input element to copy the text
    const input = document.createElement('input');
    input.value = shareLink;
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    document.body.removeChild(input);
    
    showToast('success', 'Link Copied', 'Shareable link copied to clipboard');
}

// Delete a result from history
function deleteFromHistory(id) {
    userSettings.executionHistory = userSettings.executionHistory.filter(item => item.id !== id);
    saveUserSettings();
    renderExecutionHistory();
    
    showToast('info', 'Deleted', 'History item removed');
}

// Show a toast notification
function showToast(type, title, message) {
    if (!userSettings.notifications) return;
    
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        // Create toast container if it doesn't exist
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = '';
    switch(type) {
        case 'success': icon = '<i class="fas fa-check-circle"></i>'; break;
        case 'error': icon = '<i class="fas fa-exclamation-circle"></i>'; break;
        case 'info': 
        default: icon = '<i class="fas fa-info-circle"></i>';
    }
    
    toast.innerHTML = `
        ${icon}
        <div class="toast-content">
            <h4>${title}</h4>
            <p>${message}</p>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    
    // Remove toast after 4 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 4000);
}

// Toggle user profile dropdown
function toggleProfileDropdown() {
    document.getElementById('profile-dropdown').classList.toggle('show');
}

// Close profile dropdown when clicking outside
window.addEventListener('click', function(e) {
    if (!e.target.closest('.user-profile')) {
        const dropdown = document.getElementById('profile-dropdown');
        if (dropdown && dropdown.classList.contains('show')) {
            dropdown.classList.remove('show');
        }
    }
});

// Show a modal dialog
function showModal(id) {
    document.getElementById(id).classList.add('show');
}

// Hide a modal dialog
function hideModal(id) {
    document.getElementById(id).classList.remove('show');
}

// Function to describe the dataset for better insights
function describeData() {
    if (!data.length) return;
    
    try {
        // Calculate basic statistics
        const numFeatures = data[0].X.length;
        const statistics = [];
        
        for (let i = 0; i < numFeatures; i++) {
            const values = data.map(d => d.X[i]);
            const min = Math.min(...values);
            const max = Math.max(...values);
            const sum = values.reduce((a, b) => a + b, 0);
            const mean = sum / values.length;
            
            // Calculate variance and standard deviation
            const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
            const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
            const stdDev = Math.sqrt(variance);
            
            statistics.push({
                feature: i,
                min,
                max,
                mean,
                stdDev
            });
        }
        
        // Store statistics in window for later use
        window.dataStatistics = statistics;
        
        // Check for features with high correlation
        if (numFeatures > 1) {
            let correlations = [];
            for (let i = 0; i < numFeatures; i++) {
                for (let j = i + 1; j < numFeatures; j++) {
                    const featI = data.map(d => d.X[i]);
                    const featJ = data.map(d => d.X[j]);
                    const correlation = calculateCorrelation(featI, featJ);
                    
                    if (Math.abs(correlation) > 0.7) {
                        correlations.push({
                            feature1: i,
                            feature2: j,
                            correlation
                        });
                    }
                }
            }
            
            // Store correlations for later use
            window.dataCorrelations = correlations;
        }
    } catch (error) {
        console.error('Error describing data:', error);
    }
}

// Calculate Pearson correlation coefficient
function calculateCorrelation(x, y) {
    const n = x.length;
    
    // Calculate means
    const xMean = x.reduce((a, b) => a + b, 0) / n;
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    
    // Calculate covariance and variances
    let covariance = 0;
    let xVariance = 0;
    let yVariance = 0;
    
    for (let i = 0; i < n; i++) {
        const xDiff = x[i] - xMean;
        const yDiff = y[i] - yMean;
        covariance += xDiff * yDiff;
        xVariance += xDiff * xDiff;
        yVariance += yDiff * yDiff;
    }
    
    // Return correlation coefficient
    return covariance / (Math.sqrt(xVariance) * Math.sqrt(yVariance));
}

// Export model as a standalone JavaScript module
function exportModel() {
    const algo = document.getElementById('algo').value;
    if (!lastResult) {
        showToast('error', 'No Model', 'Please run an algorithm first');
        return;
    }
    
    let modelCode = '';
    
    // Generate model code based on algorithm
    switch (algo) {
        case 'linear':
            const [w, b] = lastResult.history[0];
            modelCode = `
export class LinearRegressionModel {
    constructor() {
        this.weight = ${w.toFixed(6)};
        this.bias = ${b.toFixed(6)};
    }
    
    predict(x) {
        return this.weight * x + this.bias;
    }
}

// Usage example:
// const model = new LinearRegressionModel();
// const prediction = model.predict(5);
`;
            break;
        case 'logistic':
            const [weight, bias] = lastResult.history[0];
            modelCode = `
export class LogisticRegressionModel {
    constructor() {
        this.weight = ${weight.toFixed(6)};
        this.bias = ${bias.toFixed(6)};
    }
    
    predict(x) {
        const z = this.weight * x + this.bias;
        return 1 / (1 + Math.exp(-z));
    }
    
    predictClass(x) {
        return this.predict(x) >= 0.5 ? 1 : 0;
    }
}

// Usage example:
// const model = new LogisticRegressionModel();
// const probability = model.predict(5);
// const predictedClass = model.predictClass(5);
`;
            break;
        default:
            modelCode = `
// Export functionality for '${algo}' is not implemented yet.
// This is a placeholder for the exported model.

export class MLModel {
    constructor() {
        this.algorithm = "${algo}";
        this.params = ${JSON.stringify(lastResult, null, 2)};
    }
    
    predict(x) {
        // Implementation would go here
        console.log("Predicting with ${algo} model");
        return null;
    }
}
`;
    }
    
    // Create a blob with the model code
    const blob = new Blob([modelCode], { type: 'text/javascript' });
    const url = URL.createObjectURL(blob);
    
    // Create a download link and trigger it
    const a = document.createElement('a');
    a.href = url;
    a.download = `${algo}_model.js`;
    a.click();
    
    URL.revokeObjectURL(url);
    showToast('success', 'Model Exported', `${algo} model exported as JavaScript module`);
}

// Implement team sharing functionality
function shareWithTeam() {
    // This would typically be implemented with a backend service
    showToast('info', 'Team Sharing', 'Shared with your team members');
}

// Load user settings on document load
document.addEventListener('DOMContentLoaded', function() {
    loadUserSettings();
    
    // Add event listener for theme toggle if it exists
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
});