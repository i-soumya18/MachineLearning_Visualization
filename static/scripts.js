// DOM Elements
const themeToggle = document.createElement('button');
const algoSelect = document.getElementById('algo');
const runBtn = document.querySelector('button[onclick="fitModel()"]');
const compareBtn = document.querySelector('button[onclick="compareAlgorithms()"]');
const resetBtn = document.querySelector('button[onclick="reset()"]');

// Theme variables
const themes = {
    light: {
        '--primary': '#4361ee',
        '--primary-dark': '#3a0ca3',
        '--secondary': '#3f37c9',
        '--accent': '#4895ef',
        '--light': '#f8f9fa',
        '--dark': '#212529',
        '--text': '#495057',
        '--text-light': '#6c757d',
        '--bg': '#f5f7fa',
        '--panel-bg': '#ffffff',
        '--border': '#e9ecef',
        '--shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        '--code-bg': '#2b2b2b'
    },
    dark: {
        '--primary': '#4895ef',
        '--primary-dark': '#4361ee',
        '--secondary': '#3f37c9',
        '--accent': '#4cc9f0',
        '--light': '#212529',
        '--dark': '#f8f9fa',
        '--text': '#e9ecef',
        '--text-light': '#adb5bd',
        '--bg': '#121212',
        '--panel-bg': '#1e1e1e',
        '--border': '#343a40',
        '--shadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
        '--code-bg': '#1e1e1e'
    }
};

// Initialize the theme
let currentTheme = 'light';

// Create theme toggle button
function createThemeToggle() {
    themeToggle.id = 'theme-toggle';
    themeToggle.innerHTML = '<i class="fas fa-moon"></i> Theme';
    themeToggle.classList.add('theme-toggle');
    document.querySelector('header').appendChild(themeToggle);

    themeToggle.addEventListener('click', toggleTheme);
}

// Toggle between light and dark theme
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);

    // Update icon
    const icon = themeToggle.querySelector('i');
    icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Apply the selected theme
function applyTheme(theme) {
    const root = document.documentElement;
    const themeColors = themes[theme];

    Object.keys(themeColors).forEach(key => {
        root.style.setProperty(key, themeColors[key]);
    });

    // Additional theme-specific adjustments
    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }
}

// Initialize UI enhancements
function initUI() {
    createThemeToggle();

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        applyTheme(currentTheme);

        // Update icon based on saved theme
        const icon = themeToggle.querySelector('i');
        icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    // Add ripple effect to buttons
    document.querySelectorAll('button, label[for="upload"]').forEach(button => {
        button.addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const ripple = document.createElement('span');
            ripple.className = 'ripple';
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;

            this.appendChild(ripple);

            setTimeout(() => {
                ripple.remove();
            }, 1000);
        });
    });

    // Add hover effects to panels
    document.querySelectorAll('.params-panel, .metrics-panel, .explanation-panel, .code-panel, .data-panel, .help-panel').forEach(panel => {
        panel.addEventListener('mouseenter', () => {
            panel.style.transform = 'translateY(-5px)';
            panel.style.boxShadow = '0 10px 15px rgba(0, 0, 0, 0.1)';
        });

        panel.addEventListener('mouseleave', () => {
            panel.style.transform = '';
            panel.style.boxShadow = '';
        });
    });

    // Initialize algorithm parameters visibility
    updateAlgorithmParams();
}

// Update visible parameters based on selected algorithm
function updateAlgorithmParams() {
    const algo = algoSelect.value;

    // Hide all parameter sections first
    document.querySelectorAll('#lr-params, #k-params, #depth-params, #eps-params, #min-samples-params, #n-components-params, #classification-params').forEach(el => {
        el.classList.add('hidden');
    });

    // Show relevant parameters
    switch(algo) {
        case 'linear':
        case 'logistic':
            document.getElementById('lr-params').classList.remove('hidden');
            if (algo === 'logistic') document.getElementById('classification-params').classList.remove('hidden');
            break;
        case 'dtree':
        case 'rf':
            document.getElementById('depth-params').classList.remove('hidden');
            document.getElementById('classification-params').classList.remove('hidden');
            break;
        case 'kmeans':
            document.getElementById('k-params').classList.remove('hidden');
            break;
        case 'dbscan':
            document.getElementById('eps-params').classList.remove('hidden');
            document.getElementById('min-samples-params').classList.remove('hidden');
            break;
        case 'pca':
            document.getElementById('n-components-params').classList.remove('hidden');
            break;
    }
}

// Update parameter value displays
function updateParameter(param) {
    const valueElement = document.getElementById(`${param}-value`);
    const inputElement = document.getElementById(param);

    if (inputElement.type === 'range') {
        valueElement.textContent = parseFloat(inputElement.value).toFixed(3);
    } else {
        valueElement.textContent = inputElement.value;
    }
}

// Event Listeners
algoSelect.addEventListener('change', updateAlgorithmParams);

// Set up parameter event listeners
document.addEventListener('DOMContentLoaded', () => {
    initUI();

    document.querySelectorAll('input[type="range"], input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            const param = this.id;
            updateParameter(param);
        });
    });
});

// Add CSS for ripple effect and theme toggle
const style = document.createElement('style');
style.textContent = `
    .ripple {
        position: absolute;
        border-radius: 50%;
        background-color: rgba(255, 255, 255, 0.7);
        transform: scale(0);
        animation: ripple 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .theme-toggle {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        color: white;
        cursor: pointer;
    }
    
    .theme-toggle:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    .dark-theme .visualization-panel,
    .dark-theme .params-panel,
    .dark-theme .metrics-panel,
    .dark-theme .explanation-panel,
    .dark-theme .code-panel,
    .dark-theme .data-panel,
    .dark-theme .help-panel {
        border: 1px solid var(--border);
    }
    
    .dark-theme #code {
        border: 1px solid var(--border);
    }
    
    .dark-theme table {
        color: var(--text);
    }
    
    .dark-theme th {
        background-color: var(--primary-dark);
    }
    
    .dark-theme tr:nth-child(even) {
        background-color: #2a2a2a;
    }
    
    .dark-theme tr:hover {
        background-color: #333333;
    }
`;
document.head.appendChild(style);