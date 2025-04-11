let data = [];
let rawData = [];
let history = [];
let stepIndex = 0;
let lastResult = {};
let comparisonResults = {};

const svg = d3.select("#visualization").attr("width", 600).attr("height", 400);
const xScale = d3.scaleLinear().domain([0, 5]).range([50, 550]);
const yScale = d3.scaleLinear().domain([0, 6]).range([350, 50]);
const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

function updateViz() {
    try {
        const vizType = document.getElementById('viz-type').value;
        const plotlyDiv = document.getElementById('plotly-viz');
        const svgViz = document.getElementById('visualization');
        plotlyDiv.classList.add('hidden');
        svgViz.classList.add('hidden');

        if (vizType === 'scatter3d' && data[0]?.X.length >= 3) {
            plotlyDiv.classList.remove('hidden');
            const trace = {
                x: data.map(d => d.X[0]),
                y: data.map(d => d.X[1]),
                z: data.map(d => d.X[2]),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    size: 5,
                    color: data.map(d => d.label ? colorScale(d.label) : '#3b82f6'),
                }
            };
            Plotly.newPlot('plotly-viz', [trace], {
                margin: { t: 0, l: 0, r: 0, b: 0 },
                scene: { aspectmode: 'cube' }
            });
        } else {
            svgViz.classList.remove('hidden');
            svg.selectAll("*").remove();
            const X = data.map(d => d.X);

            if (vizType === 'scatter') {
                svg.selectAll("circle")
                    .data(data)
                    .enter()
                    .append("circle")
                    .attr("cx", d => xScale(d.X[0]))
                    .attr("cy", d => yScale(d.X[1]))
                    .attr("r", 5)
                    .attr("fill", d => d.label ? colorScale(d.label) : "#3b82f6")
                    .transition().duration(500);
            } else if (vizType === 'line') {
                const line = d3.line()
                    .x(d => xScale(d.X[0]))
                    .y(d => yScale(d.X[1]));
                svg.append("path")
                    .datum(data)
                    .attr("d", line)
                    .attr("stroke", "#3b82f6")
                    .attr("stroke-width", 2)
                    .attr("fill", "none")
                    .transition().duration(500);
            } else if (vizType === 'bar') {
                svg.selectAll("rect")
                    .data(data)
                    .enter()
                    .append("rect")
                    .attr("x", d => xScale(d.X[0]) - 5)
                    .attr("y", d => yScale(d.X[1]))
                    .attr("width", 10)
                    .attr("height", d => 400 - yScale(d.X[1]) - 50)
                    .attr("fill", d => d.label ? colorScale(d.label) : "#3b82f6")
                    .transition().duration(500);
            }
        }
    } catch (error) {
        console.error(`Visualization error: ${error}`);
        alert(`Error rendering visualization: ${error.message}`);
    }
}

document.getElementById('viz-type').addEventListener('change', updateViz);

function previewData() {
    try {
        const previewDiv = document.getElementById('data-preview');
        if (!rawData.length) {
            previewDiv.innerHTML = 'No data loaded. <button onclick="addDataRow()" class="bg-blue-600 text-white px-2 py-1 rounded-md hover:bg-blue-700">Add Row</button>';
            return;
        }
        const sample = rawData.slice(0, 5);
        let preview = '<table><tr class="bg-gray-700">' +
            Object.keys(sample[0]).map(col => `<th>${col}</th>`).join('') +
            '<th>Edit</th></tr>';
        sample.forEach((row, i) => {
            preview += '<tr>' +
                Object.values(row).map(val => `<td contenteditable="true" onblur="editData(${i}, this.parentNode.cells, '${Object.keys(row)[this.cellIndex]}')">${val}</td>`).join('') +
                `<td><button onclick="deleteDataRow(${i})" class="text-red-500 hover:text-red-700"><i class="fas fa-trash"></i></button></td></tr>`;
        });
        preview += '</table><button onclick="addDataRow()" class="mt-2 bg-blue-600 text-white px-2 py-1 rounded-md hover:bg-blue-700">Add Row</button>';
        previewDiv.innerHTML = preview;
    } catch (error) {
        console.error(`Data preview error: ${error}`);
        document.getElementById('data-preview').textContent = `Error previewing data: ${error.message}`;
    }
}

function editData(index, cells, key) {
    try {
        const newValue = cells[Array.from(cells).findIndex(c => c.textContent === rawData[index][key])].textContent;
        rawData[index][key] = isNaN(+newValue) ? newValue : +newValue;
        updateDataFromColumns();
    } catch (error) {
        console.error(`Edit data error: ${error}`);
        alert(`Error editing data: ${error.message}`);
    }
}

function deleteDataRow(index) {
    try {
        rawData.splice(index, 1);
        updateDataFromColumns();
        previewData();
    } catch (error) {
        console.error(`Delete data error: ${error}`);
        alert(`Error deleting row: ${error.message}`);
    }
}

function addDataRow() {
    try {
        const newRow = rawData.length ? Object.fromEntries(Object.keys(rawData[0]).map(k => [k, 0])) : { X1: 0, X2: 0 };
        rawData.push(newRow);
        updateDataFromColumns();
        previewData();
    } catch (error) {
        console.error(`Add data error: ${error}`);
        alert(`Error adding row: ${error.message}`);
    }
}

function updateInputSelectors() {
    try {
        const numInputs = parseInt(document.getElementById('num-inputs').value);
        const container = document.getElementById('input-selectors');
        container.innerHTML = '';
        const columns = rawData.length ? Object.keys(rawData[0]) : [];
        for (let i = 0; i < numInputs; i++) {
            const div = document.createElement('div');
            div.innerHTML = `
                <label class="block text-sm font-medium text-gray-300">Input ${i + 1}:</label>
                <select id="input-${i}" class="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-200">
                    ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                </select>
            `;
            container.appendChild(div);
            if (columns.length) document.getElementById(`input-${i}`).value = columns[i % columns.length];
        }
        updateDataFromColumns();
    } catch (error) {
        console.error(`Input selector update error: ${error}`);
        alert(`Error updating input selectors: ${error.message}`);
    }
}

document.getElementById('num-inputs').addEventListener('change', updateInputSelectors);

document.getElementById('upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 100 * 1024 * 1024) {
        alert("File size exceeds 100 MB limit");
        return;
    }
    document.getElementById('data-status').textContent = 'Data Status: Loading...';
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            rawData = d3.csvParse(event.target.result);
            if (!rawData.length) throw new Error("Empty CSV");
            const columns = Object.keys(rawData[0]);
            const labelSelect = document.getElementById('label-col');
            labelSelect.innerHTML = '<option value="">None</option>' + columns.map(col => `<option value="${col}">${col}</option>`).join('');
            updateInputSelectors();
            previewData();
            document.getElementById('data-status').textContent = 'Data Status: Loaded';
        } catch (error) {
            console.error(`CSV loading error: ${error}`);
            alert(`Error loading CSV: ${error.message}`);
            document.getElementById('data-status').textContent = `Data Status: Error - ${error.message}`;
        }
    };
    reader.readAsText(file);
});

function updateDataFromColumns() {
    try {
        const numInputs = parseInt(document.getElementById('num-inputs').value);
        const labelCol = document.getElementById('label-col').value;
        
        // First pass - detect categorical columns and create mappings
        const categoricalMappings = {};
        for (let i = 0; i < numInputs; i++) {
            const col = document.getElementById(`input-${i}`).value;
            const values = rawData.map(row => row[col]);
            
            // Check if this column contains non-numeric data
            if (values.some(v => isNaN(+v))) {
                // This is a categorical column, create a mapping
                const uniqueValues = [...new Set(values)];
                categoricalMappings[col] = Object.fromEntries(
                    uniqueValues.map((val, index) => [val, index])
                );
            }
        }
        
        // Second pass - convert data using mappings
        data = rawData.map(row => {
            const X = [];
            for (let i = 0; i < numInputs; i++) {
                const col = document.getElementById(`input-${i}`).value;
                if (categoricalMappings[col]) {
                    // Use the mapping for categorical data
                    X.push(categoricalMappings[col][row[col]]);
                } else {
                    // Regular numeric conversion
                    X.push(+row[col]);
                }
            }
            return { X, label: labelCol ? row[labelCol] : null };
        });
        
        // The check below is no longer needed since we handle categorical data
        // if (data.some(d => d.X.some(v => isNaN(v)))) throw new Error("Selected input columns contain non-numeric data");
        
        // Update scales for visualization
        xScale.domain([d3.min(data, d => d.X[0]), d3.max(data, d => d.X[0])]);
        yScale.domain([d3.min(data, d => d.X[1]), d3.max(data, d => d.X[1])]);
        colorScale.domain([...new Set(data.map(d => d.label))].filter(l => l !== null));
        updateViz();
    } catch (error) {
        console.error(`Data update error: ${error}`);
        alert(`Error updating data: ${error.message}`);
    }
}

document.getElementById('label-col').addEventListener('change', updateDataFromColumns);
document.getElementById('normalize').addEventListener('change', () => updateDataFromColumns() && fitModel());
document.getElementById('cv').addEventListener('change', fitModel);

function generateRandomData() {
    try {
        const numInputs = parseInt(document.getElementById('num-inputs').value);
        rawData = Array.from({ length: 20 }, (_, i) => {
            const row = {};
            for (let j = 0; j < numInputs; j++) row[`X${j + 1}`] = Math.random() * 5;
            return row;
        });
        data = rawData.map(row => ({ X: Object.values(row), label: null }));
        xScale.domain([0, 5]);
        yScale.domain([0, 5]);
        document.getElementById('input-selectors').innerHTML = '';
        document.getElementById('label-col').innerHTML = '<option value="">None</option>' +
            Object.keys(rawData[0]).map(col => `<option value="${col}">${col}</option>`).join('');
        updateInputSelectors();
        updateViz();
        previewData();
        document.getElementById('data-status').textContent = 'Data Status: Random data generated';
    } catch (error) {
        console.error(`Random data generation error: ${error}`);
        alert(`Error generating random data: ${error.message}`);
    }
}

document.getElementById('algo').addEventListener('change', (e) => {
    const algo = e.target.value;
    const isSupervised = ['linear', 'logistic', 'dtree', 'rf', 'svm'].includes(algo);
    document.getElementById('lr-params').classList.toggle('hidden', !['linear', 'logistic'].includes(algo));
    document.getElementById('k-params').classList.toggle('hidden', algo !== 'kmeans');
    document.getElementById('depth-params').classList.toggle('hidden', !['dtree', 'rf'].includes(algo));
    document.getElementById('eps-params').classList.toggle('hidden', algo !== 'dbscan');
    document.getElementById('min-samples-params').classList.toggle('hidden', algo !== 'dbscan');
    document.getElementById('n-components-params').classList.toggle('hidden', algo !== 'pca');
    document.getElementById('classification-params').classList.toggle('hidden', !['dtree', 'rf', 'svm'].includes(algo));
    updateCode();
    fitModel();
});

function updateParameter(id) {
    document.getElementById(`${id}-value`).textContent = document.getElementById(id).value;
    updateCode();
    fitModel();
}

['lr', 'k', 'depth', 'eps', 'min-samples', 'n-components'].forEach(id => {
    const element = document.getElementById(id);
    if (element) element.addEventListener('input', () => updateParameter(id));
});

async function fitModel() {
    const algo = document.getElementById('algo').value;
    document.getElementById('algo-status').textContent = 'Algorithm Status: Running...';
    const payload = {
        X: data.map(d => d.X),
        y: data.map(d => d.label).filter(l => l !== null && !isNaN(+l)).map(l => +l),
        labels: data.map(d => d.label),
        algo,
        normalize: document.getElementById('normalize').checked,
        cv: document.getElementById('cv').checked,
        ...(algo === 'linear' || algo === 'logistic' ? { lr: +document.getElementById('lr').value } : {}),
        ...(algo === 'kmeans' ? { k: +document.getElementById('k').value } : {}),
        ...(algo === 'dtree' || algo === 'rf' ? { depth: +document.getElementById('depth').value, is_classification: document.getElementById('is_classification').checked } : {}),
        ...(algo === 'svm' ? { is_classification: document.getElementById('is_classification').checked } : {}),
        ...(algo === 'dbscan' ? { eps: +document.getElementById('eps').value, min_samples: +document.getElementById('min-samples').value } : {}),
        ...(algo === 'pca' ? { n_components: +document.getElementById('n-components').value } : {})
    };

    try {
        const response = await fetch('/fit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error((await response.json()).error || 'Request failed');
        const result = await response.json();
        lastResult = result;
        history = result.history || result.boundaries || result.transformed || [];
        stepIndex = 0;
        document.getElementById('step-btn').disabled = !result.history;
        if (['dtree', 'rf', 'svm'].includes(algo)) visualizeBoundaries(result.boundaries);
        else if (algo === 'kmeans') visualizeClusters(result.history, result.clusters);
        else if (algo === 'dbscan') visualizeClusters([], result.clusters);
        else if (algo === 'pca') visualizePCA(result.transformed, result.variance);
        else stepThrough();

        let metricsText = '';
        if (result.metrics) {
            if ('mse' in result.metrics) metricsText += `MSE: ${result.metrics.mse.toFixed(4)}, R²: ${result.metrics.r2.toFixed(4)}`;
            if ('accuracy' in result.metrics) metricsText += `Accuracy: ${(result.metrics.accuracy * 100).toFixed(2)}%`;
            if ('silhouette' in result.metrics) metricsText += `Silhouette Score: ${result.metrics.silhouette.toFixed(4)}`;
            if ('cv_mse' in result.metrics) metricsText += `\nCV MSE: ${result.metrics.cv_mse.toFixed(4)}, CV R²: ${result.metrics.cv_r2.toFixed(4)}`;
            if ('cv_accuracy' in result.metrics) metricsText += `\nCV Accuracy: ${(result.metrics.cv_accuracy * 100).toFixed(2)}%`;
        } else if (algo === 'pca') {
            metricsText = `Explained Variance: ${result.variance.map(v => v.toFixed(4)).join(', ')}`;
        }
        document.getElementById('metrics').textContent = metricsText;
        document.getElementById('algo-status').textContent = 'Algorithm Status: Completed';
    } catch (error) {
        console.error(`Fit model error: ${error}`);
        alert(`Error running algorithm: ${error.message}`);
        document.getElementById('algo-status').textContent = `Algorithm Status: Error - ${error.message}`;
    }
}

function stepThrough() {
    if (stepIndex >= history.length) {
        document.getElementById('step-btn').disabled = true;
        return;
    }
    const algo = document.getElementById('algo').value;
    svg.selectAll(".fit").remove();

    try {
        if (algo === 'linear' || algo === 'logistic') {
            const [w, b] = history[stepIndex];
            const line = d3.line()
                .x(d => xScale(d[0]))
                .y(d => yScale(algo === 'logistic' ? 1 / (1 + Math.exp(-d[1])) : d[1]));
            const lineData = [
                [xScale.domain()[0], w * xScale.domain()[0] + b],
                [xScale.domain()[1], w * xScale.domain()[1] + b]
            ];
            svg.append("path")
                .attr("class", "fit")
                .attr("d", line(lineData))
                .attr("stroke", algo === 'linear' ? "#ef4444" : "#9333ea")
                .attr("stroke-width", 2)
                .transition().duration(500);
            document.getElementById('explanation').textContent =
                `Step ${stepIndex + 1}: Weight = ${w.toFixed(2)}, Bias = ${b.toFixed(2)}.`;
        } else if (algo === 'kmeans') {
            const centroids = history[stepIndex];
            svg.selectAll(".centroid")
                .data(centroids)
                .enter()
                .append("circle")
                .attr("class", "fit")
                .attr("cx", d => xScale(d[0]))
                .attr("cy", d => yScale(d[1]))
                .attr("r", 8)
                .attr("fill", "#ef4444")
                .transition().duration(500);
            document.getElementById('explanation').textContent =
                `Step ${stepIndex + 1}: Centroids moving.`;
        }
        stepIndex++;
    } catch (error) {
        console.error(`Step-through error: ${error}`);
        alert(`Error stepping through: ${error.message}`);
    }
}

function visualizeBoundaries(boundaries) {
    svg.selectAll(".fit").remove();
    svg.selectAll(".boundary")
        .data(boundaries)
        .enter()
        .append("rect")
        .attr("class", "fit")
        .attr("x", d => xScale(d[0]))
        .attr("y", d => yScale(d[1]))
        .attr("width", xScale(0.1) - xScale(0))
        .attr("height", yScale(0) - yScale(0.1))
        .attr("fill", d => d3.interpolateRainbow(d[2] / (document.getElementById('is_classification').checked ? 1 : d3.max(boundaries, b => b[2]))))
        .attr("opacity", 0.5)
        .transition().duration(500);
    document.getElementById('explanation').textContent = "Boundaries colored by prediction.";
}

function visualizeClusters(history, clusters) {
    svg.selectAll(".fit").remove();
    clusters.forEach((cluster, i) => {
        svg.selectAll(`.cluster-${i}`)
            .data(cluster)
            .enter()
            .append("circle")
            .attr("class", "fit")
            .attr("cx", d => xScale(d[0]))
            .attr("cy", d => yScale(d[1]))
            .attr("r", 5)
            .attr("fill", colorScale(i))
            .transition().duration(500);
    });
    if (history.length) {
        svg.selectAll(".centroid")
            .data(history[0])
            .enter()
            .append("circle")
            .attr("class", "fit")
            .attr("cx", d => xScale(d[0]))
            .attr("cy", d => yScale(d[1]))
            .attr("r", 8)
            .attr("fill", "#ef4444")
            .transition().duration(500);
    }
    document.getElementById('explanation').textContent = `Clusters formed: ${clusters.length}`;
}

function visualizePCA(transformed, variance) {
    const vizType = document.getElementById('viz-type').value;
    const plotlyDiv = document.getElementById('plotly-viz');
    const svgViz = document.getElementById('visualization');
    plotlyDiv.classList.add('hidden');
    svgViz.classList.add('hidden');

    if (vizType === 'scatter3d' && transformed[0].length >= 3) {
        plotlyDiv.classList.remove('hidden');
        const trace = {
            x: transformed.map(d => d[0]),
            y: transformed.map(d => d[1]),
            z: transformed.map(d => d[2]),
            mode: 'markers',
            type: 'scatter3d',
            marker: { size: 5, color: '#3b82f6' }
        };
        Plotly.newPlot('plotly-viz', [trace], { margin: { t: 0, l: 0, r: 0, b: 0 }, scene: { aspectmode: 'cube' } });
    } else {
        svgViz.classList.remove('hidden');
        svg.selectAll("*").remove();
        svg.selectAll("circle")
            .data(transformed)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d[0]))
            .attr("cy", d => yScale(d[1]))
            .attr("r", 5)
            .attr("fill", "#3b82f6")
            .transition().duration(500);
    }
    document.getElementById('explanation').textContent = "Data projected onto principal components.";
}

function updateCode() {
    const algo = document.getElementById('algo').value;
    let code = '';
    if (algo === 'linear') code = `from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X, y)`;
    else if (algo === 'logistic') code = `from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(C=${1/document.getElementById('lr').value}, max_iter=100)\nmodel.fit(X, y)`;
    else if (algo === 'dtree') code = `from sklearn.tree import ${document.getElementById('is_classification').checked ? 'DecisionTreeClassifier' : 'DecisionTreeRegressor'}\nmodel = ${document.getElementById('is_classification').checked ? 'DecisionTreeClassifier' : 'DecisionTreeRegressor'}(max_depth=${document.getElementById('depth').value})\nmodel.fit(X, y)`;
    else if (algo === 'rf') code = `from sklearn.ensemble import ${document.getElementById('is_classification').checked ? 'RandomForestClassifier' : 'RandomForestRegressor'}\nmodel = ${document.getElementById('is_classification').checked ? 'RandomForestClassifier' : 'RandomForestRegressor'}(max_depth=${document.getElementById('depth').value}, n_estimators=10)\nmodel.fit(X, y)`;
    else if (algo === 'svm') code = `from sklearn.svm import ${document.getElementById('is_classification').checked ? 'SVC' : 'SVR'}\nmodel = ${document.getElementById('is_classification').checked ? 'SVC' : 'SVR'}(kernel='rbf')\nmodel.fit(X, y)`;
    else if (algo === 'kmeans') code = `from sklearn.cluster import KMeans\nmodel = KMeans(n_clusters=${document.getElementById('k').value})\nmodel.fit(X)`;
    else if (algo === 'dbscan') code = `from sklearn.cluster import DBSCAN\nmodel = DBSCAN(eps=${document.getElementById('eps').value}, min_samples=${document.getElementById('min-samples').value})\nmodel.fit(X)`;
    else if (algo === 'pca') code = `from sklearn.decomposition import PCA\nmodel = PCA(n_components=${document.getElementById('n-components').value})\nmodel.fit_transform(X)`;
    document.getElementById('code').textContent = code;
}

async function loadScenario() {
    try {
        document.getElementById('data-status').textContent = 'Data Status: Loading scenario...';
        const response = await fetch('/scenario');
        if (!response.ok) throw new Error((await response.json()).error);
        rawData = (await response.json());
        data = rawData.map(d => ({ X: [d.SepalLengthCm, d.SepalWidthCm, d.PetalLengthCm, d.PetalWidthCm], label: d.Species }));
        xScale.domain([d3.min(data, d => d.X[0]), d3.max(data, d => d.X[0])]);
        yScale.domain([d3.min(data, d => d.X[1]), d3.max(data, d => d.X[1])]);
        colorScale.domain([...new Set(data.map(d => d.label))]);
        updateViz();
        previewData();
        document.getElementById('algo').value = 'kmeans';
        document.getElementById('explanation').textContent = 'Scenario: Cluster Iris flowers.';
        updateCode();

        const columns = Object.keys(rawData[0]);
        const labelSelect = document.getElementById('label-col');
        labelSelect.innerHTML = '<option value="">None</option>' + columns.map(col => `<option value="${col}">${col}</option>`).join('');
        labelSelect.value = 'Species';
        document.getElementById('num-inputs').value = 4;
        updateInputSelectors();
        document.getElementById('data-status').textContent = 'Data Status: Scenario loaded';
    } catch (error) {
        console.error(`Scenario loading error: ${error}`);
        alert(`Error loading scenario: ${error.message}`);
        document.getElementById('data-status').textContent = `Data Status: Error - ${error.message}`;
    }
}

async function compareAlgorithms() {
    const algorithms = ['linear', 'logistic', 'dtree', 'rf', 'svm', 'kmeans', 'dbscan', 'pca'];
    comparisonResults = {};
    document.getElementById('algo-status').textContent = 'Algorithm Status: Comparing...';

    try {
        for (const algo of algorithms) {
            const payload = {
                X: data.map(d => d.X),
                y: data.map(d => d.label).filter(l => l !== null && !isNaN(+l)).map(l => +l),
                labels: data.map(d => d.label),
                algo,
                normalize: document.getElementById('normalize').checked,
                cv: document.getElementById('cv').checked,
                ...(algo === 'linear' || algo === 'logistic' ? { lr: +document.getElementById('lr').value } : {}),
                ...(algo === 'kmeans' ? { k: +document.getElementById('k').value } : {}),
                ...(algo === 'dtree' || algo === 'rf' ? { depth: +document.getElementById('depth').value, is_classification: document.getElementById('is_classification').checked } : {}),
                ...(algo === 'svm' ? { is_classification: document.getElementById('is_classification').checked } : {}),
                ...(algo === 'dbscan' ? { eps: +document.getElementById('eps').value, min_samples: +document.getElementById('min-samples').value } : {}),
                ...(algo === 'pca' ? { n_components: +document.getElementById('n-components').value } : {})
            };

            const response = await fetch('/fit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) continue;
            comparisonResults[algo] = await response.json();
        }

        let comparisonText = 'Comparison Results:\n';
        for (const [algo, result] of Object.entries(comparisonResults)) {
            comparisonText += `${algo}: `;
            if (result.metrics) {
                if ('mse' in result.metrics) comparisonText += `MSE: ${result.metrics.mse.toFixed(4)}, R²: ${result.metrics.r2.toFixed(4)}`;
                if ('accuracy' in result.metrics) comparisonText += `Accuracy: ${(result.metrics.accuracy * 100).toFixed(2)}%`;
                if ('silhouette' in result.metrics) comparisonText += `Silhouette: ${result.metrics.silhouette.toFixed(4)}`;
                if ('cv_mse' in result.metrics) comparisonText += `, CV MSE: ${result.metrics.cv_mse.toFixed(4)}, CV R²: ${result.metrics.cv_r2.toFixed(4)}`;
                if ('cv_accuracy' in result.metrics) comparisonText += `, CV Accuracy: ${(result.metrics.cv_accuracy * 100).toFixed(2)}%`;
            } else if (algo === 'pca') {
                comparisonText += `Variance: ${result.variance.map(v => v.toFixed(4)).join(', ')}`;
            }
            comparisonText += '\n';
        }
        document.getElementById('metrics').textContent = comparisonText;
        document.getElementById('explanation').textContent = 'Comparison of all algorithms completed.';
        document.getElementById('algo-status').textContent = 'Algorithm Status: Comparison Completed';
    } catch (error) {
        console.error(`Comparison error: ${error}`);
        alert(`Error comparing algorithms: ${error.message}`);
        document.getElementById('algo-status').textContent = `Algorithm Status: Error - ${error.message}`;
    }
}

async function runGridSearch() {
    const algo = document.getElementById('algo').value;
    document.getElementById('algo-status').textContent = 'Algorithm Status: Running Grid Search...';

    const paramGrid = {};
    if (algo === 'linear') paramGrid.fit_intercept = [true, false];
    else if (algo === 'logistic') paramGrid.C = [0.1, 1, 10];
    else if (algo === 'dtree') paramGrid.max_depth = [3, 5, 7];
    else if (algo === 'rf') paramGrid.max_depth = [3, 5, 7];
    else if (algo === 'svm') paramGrid.C = [0.1, 1, 10];
    else if (algo === 'kmeans') paramGrid.n_clusters = [2, 3, 4];
    else if (algo === 'pca') paramGrid.n_components = [2, 3, 4];
    else if (algo === 'dbscan') {
        document.getElementById('algo-status').textContent = 'Algorithm Status: Grid Search not supported for DBSCAN';
        return;
    }
    paramGrid.is_classification = document.getElementById('is_classification').checked;

    const payload = {
        X: data.map(d => d.X),
        y: data.map(d => d.label).filter(l => l !== null && !isNaN(+l)).map(l => +l),
        labels: data.map(d => d.label),
        algo,
        param_grid: paramGrid
    };

    try {
        const response = await fetch('/grid_search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error((await response.json()).error || 'Request failed');
        const result = await response.json();
        document.getElementById('metrics').textContent = `Best Params: ${JSON.stringify(result.best_params)}\nBest Score: ${result.best_score.toFixed(4)}`;
        document.getElementById('explanation').textContent = `Grid Search completed for ${algo}.`;
        document.getElementById('algo-status').textContent = 'Algorithm Status: Grid Search Completed';
    } catch (error) {
        console.error(`Grid search error: ${error}`);
        alert(`Error running grid search: ${error.message}`);
        document.getElementById('algo-status').textContent = `Algorithm Status: Error - ${error.message}`;
    }
}

function downloadViz() {
    try {
        const vizType = document.getElementById('viz-type').value;
        if (vizType === 'scatter3d') {
            Plotly.downloadImage('plotly-viz', { format: 'png', width: 600, height: 400, filename: 'ml_visualization' });
        } else {
            const svgData = new XMLSerializer().serializeToString(document.getElementById('visualization'));
            const blob = new Blob([svgData], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ml_visualization.svg';
            a.click();
            URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error(`Download error: ${error}`);
        alert(`Error downloading visualization: ${error.message}`);
    }
}

function exportResults() {
    try {
        const algo = document.getElementById('algo').value;
        let csvContent = "data:text/csv;charset=utf-8,";
        if (algo === 'linear' || algo === 'logistic') {
            csvContent += "Weight,Bias\n";
            history.forEach(row => csvContent += `${row[0]},${row[1]}\n`);
        } else if (['dtree', 'rf', 'svm'].includes(algo)) {
            csvContent += "X,Y,Prediction\n";
            history.forEach(row => csvContent += `${row[0]},${row[1]},${row[2]}\n`);
        } else if (algo === 'kmeans' || algo === 'dbscan') {
            csvContent += "Cluster,X,Y\n";
            lastResult.clusters.forEach((cluster, i) => {
                cluster.forEach(point => csvContent += `${i},${point[0]},${point[1]}\n`);
            });
        } else if (algo === 'pca') {
            csvContent += "PC1,PC2,PC3\n";
            history.forEach(row => csvContent += `${row[0]},${row[1]},${row[2] || ''}\n`);
        }
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `${algo}_results.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        console.error(`Export error: ${error}`);
        alert(`Error exporting results: ${error.message}`);
    }
}

function reset() {
    try {
        data = [];
        rawData = [];
        xScale.domain([0, 5]);
        yScale.domain([0, 6]);
        history = [];
        stepIndex = 0;
        lastResult = {};
        comparisonResults = {};
        document.getElementById('step-btn').disabled = true;
        document.getElementById('algo').value = 'linear';
        document.getElementById('lr').value = '0.01';
        document.getElementById('lr-value').textContent = '0.01';
        document.getElementById('k').value = '2';
        document.getElementById('k-value').textContent = '2';
        document.getElementById('depth').value = '3';
        document.getElementById('depth-value').textContent = '3';
        document.getElementById('eps').value = '0.5';
        document.getElementById('eps-value').textContent = '0.5';
        document.getElementById('min-samples').value = '5';
        document.getElementById('min-samples-value').textContent = '5';
        document.getElementById('n-components').value = '2';
        document.getElementById('n-components-value').textContent = '2';
        document.getElementById('is_classification').checked = false;
        document.getElementById('normalize').checked = false;
        document.getElementById('cv').checked = false;
        document.getElementById('num-inputs').value = '2';
        document.getElementById('input-selectors').innerHTML = '';
        document.getElementById('label-col').innerHTML = '<option value="">None</option>';
        document.getElementById('viz-type').value = 'scatter';
        updateViz();
        previewData();
        updateCode();
        document.getElementById('data-status').textContent = 'Data Status: No data loaded';
        document.getElementById('algo-status').textContent = 'Algorithm Status: Not running';
        document.getElementById('metrics').textContent = '';
        document.getElementById('explanation').textContent = '';
    } catch (error) {
        console.error(`Reset error: ${error}`);
        alert(`Error resetting: ${error.message}`);
    }
}

updateViz();
previewData();
describeData();
updateCode();