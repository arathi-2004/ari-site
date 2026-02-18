document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const diseaseNameSpan = document.getElementById('disease-name');
    const confidenceScoreSpan = document.getElementById('confidence-score');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');

    // Handle file selection
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                analyzeBtn.classList.remove('hidden');
                resultSection.classList.add('hidden'); // Hide previous results
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle analysis button click
    analyzeBtn.addEventListener('click', () => {
        const file = fileInput.files[0];
        if (!file) return;

        // Show loader, hide previous results
        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                loader.classList.add('hidden');
                resultContent.classList.remove('hidden');

                if (data.error) {
                    diseaseNameSpan.textContent = "തകരാർ: " + data.error;
                    confidenceScoreSpan.textContent = "-";
                } else {
                    diseaseNameSpan.textContent = data.class;
                    confidenceScoreSpan.textContent = data.confidence.toFixed(2);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                loader.classList.add('hidden');
                resultContent.classList.remove('hidden');
                diseaseNameSpan.textContent = "സെർവറുമായി ബന്ധപ്പെടാൻ സാധിക്കുന്നില്ല.";
            });
    });
});
