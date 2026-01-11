// Get form elements
const form = document.getElementById('predictionForm');
const descriptionInput = document.getElementById('description');
const charCount = document.getElementById('charCount');
const submitBtn = document.getElementById('submitBtn');
const resultContainer = document.getElementById('resultContainer');
const errorContainer = document.getElementById('errorContainer');
const resultsDiv = document.getElementById('results');
const errorMessage = document.getElementById('errorMessage');

// Update character count
descriptionInput.addEventListener('input', () => {
    charCount.textContent = descriptionInput.value.length;
});

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results/errors
    resultContainer.style.display = 'none';
    errorContainer.style.display = 'none';
    
    // Get form values
    const description = descriptionInput.value.trim();
    const threshold = parseFloat(document.getElementById('threshold').value) || 0.55;
    const topK = parseInt(document.getElementById('topK').value) || 3;
    
    if (!description) {
        showError('Please enter a movie description');
        return;
    }
    
    // Disable submit button and show loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline';
    
    try {
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                description: description,
                threshold: threshold,
                top_k: topK
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while making the prediction. Please try again.');
    } finally {
        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

function displayResults(data) {
    const { prediction, threshold, top_k } = data;
    const genres = prediction.genres || [];
    
    let html = '';
    
    if (genres.length === 0) {
        html = '<p style="color: #888;">No genres predicted above the threshold.</p>';
    } else {
        html = '<div class="genre-list">';
        genres.forEach(genre => {
            html += `<span class="genre-tag">${genre}</span>`;
        });
        html += '</div>';
    }
    
    html += `<div class="result-info">
        <p><strong>Genres Found:</strong> ${prediction.genre_count}</p>
        <p><strong>Threshold:</strong> ${threshold}</p>
        <p><strong>Top K:</strong> ${top_k}</p>
    </div>`;
    
    resultsDiv.innerHTML = html;
    resultContainer.style.display = 'block';
    
    // Scroll to results
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorContainer.style.display = 'block';
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
