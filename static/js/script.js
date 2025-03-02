function previewFile() {
    var preview = document.getElementById('image-preview');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
        preview.style.display = 'none';
    }
}

function showSpeechBox() {
    var label = document.getElementById('speechUploadLabel');
}

document.addEventListener("DOMContentLoaded", function () {
    const textElement = document.getElementById("typing-text");
    const text = "Analyse Sentiments from Image📷 | Speech🎙️ | Text📝 ";
    let index = 0;

    function type() {
        if (index < text.length) {
            textElement.innerHTML += text.charAt(index);
            index++;
            setTimeout(type, 30);
        } else {
            textElement.style.borderRight = "none";
        }
    }

    type();

    // Chart.js Integration for Confidence Graph
    if (document.getElementById('confidenceChart')) {
        var confidenceScores = JSON.parse(document.getElementById('confidenceChart').getAttribute('data-scores'));
        var labels = JSON.parse(document.getElementById('confidenceChart').getAttribute('data-labels'));
        var ctx = document.getElementById('confidenceChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence Level',
                    data: confidenceScores,
                    backgroundColor: 'rgba(225, 167, 0, 0.5)',
                    borderColor: 'rgba(215, 178, 102, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Confidence Levels for Predicted Emotion', // Graph Title
                        font: {
                            size: 18
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Emotions', // X-axis Label
                            font: {
                                size: 14
                            }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Confidence Score', // Y-axis Label
                            font: {
                                size: 14
                            }
                        }
                    }
                }
            }
        });
        // Resize the chart container on small screens
        function adjustChartSize() {
            var chartContainer = document.getElementById('chart-container');
            if (window.innerWidth < 768) {
                chartContainer.style.width = "80%";
                chartContainer.style.height = "300px";
            } else {
                chartContainer.style.width = "600px";
                chartContainer.style.height = "400px";
            }
        }

        adjustChartSize();
        window.addEventListener('resize', adjustChartSize);
    }
});