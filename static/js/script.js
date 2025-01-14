document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('choosefile');
    const file = fileInput.files[0];

    const selectElement = document.getElementById('model-select');
    const selectedModel = selectElement.value; // Get the selected value

    if (!file) {
        alert("Please select a file before submitting.");
        return;
    }

    if (!selectedModel) {
        alert("Please select a model before submitting.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', selectedModel);

    // Show loading indicator
    document.querySelector('.loading').style.display = 'block';
    document.getElementById('results').style.opacity = 0;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to upload file');
        }

        const result = await response.json();

        // Display predictions in a table format
        const predictionsList = document.getElementById('predictions-list');
        predictionsList.innerHTML = ''; // Clear previous content

        // Create a table element
        const table = document.createElement('table');
        table.classList.add('predictions-table'); // Add a CSS class for styling

        // Add table headers
        const headers = ['Window', 'Prediction'];
        const headerRow = document.createElement('tr');
        headers.forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        // Sort predictions by window number
        const sortedEntries = Object.entries(result).sort((a, b) => {
            // Extract numerical part of the window key for comparison
            const numA = parseInt(a[0].match(/\d+/));
            const numB = parseInt(b[0].match(/\d+/));
            return numA - numB;
        });

        // Add table rows for each sorted prediction
        for (const [window, prediction] of sortedEntries) {
            const row = document.createElement('tr');
            row.classList.add('prediction-row');
            row.dataset.window = window;

            const windowCell = document.createElement('td');
            windowCell.textContent = window;
            row.appendChild(windowCell);

            const predictionCell = document.createElement('td');
            predictionCell.textContent = prediction;
            row.appendChild(predictionCell);

            table.appendChild(row);

            // Add click event listener to handle window selection
            row.addEventListener('click', async () => {
                await handleWindowSelection(window);
            });
        }

        predictionsList.appendChild(table);

        document.querySelector('.loading').style.display = 'none';
        document.getElementById('results').style.opacity = 1;

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the file.');
    }
});

async function handleWindowSelection(window) {
    try {
        const response = await fetch(`/window-details?window=${window}`, {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error('Failed to fetch window details');
        }

        const data = await response.json();

        // Show the selected window details section
        document.getElementById('selected-window').style.display = 'block';

        // Update the audio playback
        const audioElement = document.getElementById('window-audio');
        audioElement.src = data.audio_url; // URL of the sliced audio
        audioElement.load();

        // Update the features list
        const featureList = document.getElementById('feature-list');
        featureList.innerHTML = ''; // Clear previous features

        for (const [feature, value] of Object.entries(data.features)) {
            const listItem = document.createElement('li');
            listItem.textContent = `${feature}: ${value}`;
            featureList.appendChild(listItem);
        }
    } catch (error) {
        console.error('Error fetching window details:', error);
        alert('Failed to fetch details for the selected window.');
    }
}

