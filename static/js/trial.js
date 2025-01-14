document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('audioFile');
    const modelSelect = document.getElementById('model-select');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const predictionsTable = document.getElementById('predictions-table');
    const windowNumber = document.getElementById('window-number');
    const viewFeaturesBtn = document.getElementById('view-features-btn');
    const featuresDisplay = document.getElementById('features-display');

    // Update file input label with selected filename
    fileInput.addEventListener('change', function () {
        const label = this.nextElementSibling;
        label.textContent = this.files[0] ? this.files[0].name : 'Choose Audio File';
    });

    // Handle form submission for predictions
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const file = fileInput.files[0];
        const model = modelSelect.value;

        if (!file || !model) {
            showError('Please select both a file and a model.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', model);

        try {
            showLoading();
            hideError();
            hideResults();

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prediction failed');
            }

            const predictions = await response.json();
            displayPredictions(predictions);
            showResults();
        } catch (error) {
            showError(error.message || 'An error occurred while processing the audio file.');
        } finally {
            hideLoading();
        }
    });

    // Handle feature visualization request
    viewFeaturesBtn.addEventListener('click', async function () {
        const window = parseInt(windowNumber.value);

        if (isNaN(window) || window < 0) {
            showError('Please enter a valid window number.');
            return;
        }

        try {
            showLoading();
            hideError();

            const response = await fetch('/features', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ window })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Feature extraction failed');
            }

            const features = await response.json();
            displayFeatures(features);
            featuresDisplay.classList.remove('hidden');
        } catch (error) {
            showError(error.message || 'Failed to load features for the selected window.');
        } finally {
            hideLoading();
        }
    });

    // Helper functions
    function showLoading() {
        loading.classList.remove('hidden');
    }

    function hideLoading() {
        loading.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }

    function showResults() {
        results.classList.remove('hidden');
    }

    function hideResults() {
        results.classList.add('hidden');
    }

    function displayPredictions(predictions) {
        const table = document.createElement('table');
        table.classList.add('predictions-table');

        // Create header
        const header = table.createTHead();
        const headerRow = header.insertRow();
        ['Window', 'Prediction'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            headerRow.appendChild(th);
        });

        // Create body
        const tbody = table.createTBody();
        Object.entries(predictions).forEach(([window, prediction]) => {
            const row = tbody.insertRow();
            const windowCell = row.insertCell();
            const predictionCell = row.insertCell();

            windowCell.textContent = window;
            predictionCell.textContent = prediction;
        });

        // Replace existing table
        predictionsTable.innerHTML = '';
        predictionsTable.appendChild(table);
    }

    function displayFeatures(features) {
        Object.entries(features).forEach(([feature, imageData]) => {
            const container = document.getElementById(`${feature}-image`);
            container.innerHTML = `<img src="data:image/png;base64,${imageData}" alt="${feature} visualization">`;
        });
    }
});
