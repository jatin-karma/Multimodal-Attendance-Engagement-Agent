/**
 * Multimodal Attendance & Engagement Agent
 * Core JavaScript Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Multimodal Attendance & Engagement Agent initialized.');

    // ----------------------------------------
    // File Upload Handler
    // ----------------------------------------
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');

    if (fileInput) {
        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                handleFileUpload(file);
            }
        });
    }

    // Drag and drop support
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#6c63ff';
            uploadArea.style.background = 'rgba(108, 99, 255, 0.05)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            uploadArea.style.background = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            uploadArea.style.background = '';
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFileUpload(file);
            }
        });
    }

    // ----------------------------------------
    // Health Check
    // ----------------------------------------
    checkSystemHealth();
});

/**
 * Handle file upload via FormData and fetch API.
 */
async function handleFileUpload(file) {
    const allowedTypes = [
        'video/mp4', 'video/avi', 'video/quicktime', 'video/webm',
        'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg'
    ];

    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm|wav|mp3|flac|ogg|m4a)$/i)) {
        showNotification('Unsupported file type. Please upload a video or audio file.', 'error');
        return;
    }

    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.innerHTML = `
            <div class="upload-icon">⏳</div>
            <p>Uploading ${file.name}...</p>
            <div class="progress-bar"><div class="progress-fill" id="upload-progress"></div></div>
        `;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showNotification(`File "${file.name}" uploaded successfully!`, 'success');
        } else {
            showNotification(result.error || 'Upload failed.', 'error');
        }
    } catch (error) {
        showNotification('Upload failed. Please try again.', 'error');
        console.error('Upload error:', error);
    }
}

/**
 * Check system health status.
 */
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('System health:', data.status);
    } catch (error) {
        console.warn('Health check failed:', error.message);
    }
}

/**
 * Display a notification toast.
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '10px',
        color: '#fff',
        fontFamily: "'Inter', sans-serif",
        fontSize: '0.9rem',
        fontWeight: '500',
        zIndex: '9999',
        opacity: '0',
        transform: 'translateX(100%)',
        transition: 'all 0.3s ease',
        maxWidth: '400px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
    });

    const colors = {
        success: 'linear-gradient(135deg, #00d4aa, #00b894)',
        error: 'linear-gradient(135deg, #ff6b6b, #ee5a24)',
        info: 'linear-gradient(135deg, #6c63ff, #5a54e0)'
    };

    notification.style.background = colors[type] || colors.info;

    document.body.appendChild(notification);

    requestAnimationFrame(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    });

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}
