// static/js/main.js
/* Graphura Portfolio Scorer - Main JavaScript */

// Global variables
let currentComparisonResults = null;

// Utility Functions
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '1100';
        document.body.appendChild(container);
    }
    
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0 mb-2" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    const toastContainerEl = document.getElementById('toast-container');
    toastContainerEl.insertAdjacentHTML('beforeend', toastHtml);
    const toastEl = toastContainerEl.lastElementChild;
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
    
    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function truncateText(text, maxLength = 50) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function getScoreColor(score) {
    if (score >= 80) return 'success';
    if (score >= 50) return 'warning';
    return 'danger';
}

function getScoreClass(score) {
    if (score >= 80) return 'score-high';
    if (score >= 50) return 'score-mid';
    return 'score-low';
}

function getReadinessBadge(readiness) {
    let badgeClass = 'bg-secondary';
    if (readiness === 'Job Ready') badgeClass = 'bg-success';
    else if (readiness === 'Almost Ready') badgeClass = 'bg-warning text-dark';
    else if (readiness === 'Needs Improvement') badgeClass = 'bg-danger';
    
    return `<span class="badge ${badgeClass}">${readiness}</span>`;
}

// File Upload Handlers
function setupDragAndDrop(dropzoneId, fileInputId, callback) {
    const dropzone = document.getElementById(dropzoneId);
    const fileInput = document.getElementById(fileInputId);
    
    if (!dropzone) return;
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('border-primary', 'bg-light');
    });
    
    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-primary', 'bg-light');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-primary', 'bg-light');
        const files = e.dataTransfer.files;
        if (files.length > 0 && callback) {
            callback(files[0]);
        }
    });
    
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0 && callback) {
                callback(e.target.files[0]);
            }
        });
    }
}

// Chart Initialization Helpers
function createDoughnutChart(ctx, labels, data, colors) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createBarChart(ctx, labels, data, label, colors = '#0d6efd') {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        drawBorder: false
                    }
                }
            }
        }
    });
}

function createHorizontalBarChart(ctx, labels, data, label, colors = '#0d6efd') {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Export functions for use in other scripts
window.showToast = showToast;
window.formatDate = formatDate;
window.truncateText = truncateText;
window.getScoreColor = getScoreColor;
window.getScoreClass = getScoreClass;
window.getReadinessBadge = getReadinessBadge;
window.setupDragAndDrop = setupDragAndDrop;
window.createDoughnutChart = createDoughnutChart;
window.createBarChart = createBarChart;
window.createHorizontalBarChart = createHorizontalBarChart;