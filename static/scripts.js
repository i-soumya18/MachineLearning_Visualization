// DOM Elements
const themeToggle = document.createElement('button');
const algoSelect = document.getElementById('algo');
const runBtn = document.querySelector('button[onclick="fitModel()"]');
const compareBtn = document.querySelector('button[onclick="compareAlgorithms()"]');
const resetBtn = document.querySelector('button[onclick="reset()"]');
const header = document.querySelector('header');
const panels = document.querySelectorAll('.control-panel, .visualization-panel, .params-panel, .metrics-panel, .explanation-panel, .code-panel, .data-panel, .help-panel');

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
    themeToggle.setAttribute('aria-label', 'Toggle theme');
    document.querySelector('header').appendChild(themeToggle);

    themeToggle.addEventListener('click', toggleTheme);
}

// Toggle between light and dark theme
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);

    // Update icon and accessibility
    const icon = themeToggle.querySelector('i');
    icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    themeToggle.setAttribute('aria-label', currentTheme === 'light' ? 'Switch to dark theme' : 'Switch to light theme');
}

// Apply the selected theme
function applyTheme(theme) {
    const root = document.documentElement;
    const themeColors = themes[theme];

    Object.keys(themeColors).forEach(key => {
        root.style.setProperty(key, themeColors[key]);
    });

    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }
}

// Add collapsible functionality to panels
function makePanelsCollapsible() {
    panels.forEach(panel => {
        const header = panel.querySelector('h2, h3');
        if (!header) return;

        // Add collapse toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
        toggleBtn.classList.add('collapse-toggle');
        toggleBtn.setAttribute('aria-label', `Toggle ${header.textContent} panel`);
        header.appendChild(toggleBtn);

        toggleBtn.addEventListener('click', () => {
            const content = panel.querySelector('div');
            const isCollapsed = content.classList.toggle('collapsed');
            toggleBtn.querySelector('i').className = isCollapsed ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
            panel.setAttribute('aria-expanded', !isCollapsed);
        });

        // Set initial ARIA attribute
        panel.setAttribute('aria-expanded', 'true');
    });
}

// Initialize UI enhancements
function initUI() {
    createThemeToggle();
    makePanelsCollapsible();

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        applyTheme(currentTheme);
        const icon = themeToggle.querySelector('i');
        icon.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        themeToggle.setAttribute('aria-label', currentTheme === 'light' ? 'Switch to dark theme' : 'Switch to light theme');
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
    panels.forEach(panel => {
        panel.addEventListener('mouseenter', () => {
            panel.style.transform = 'translateY(-5px)';
            panel.style.boxShadow = '0 10px 15px rgba(0, 0, 0, 0.1)';
        });

        panel.addEventListener('mouseleave', () => {
            panel.style.transform = '';
            panel.style.boxShadow = '';
        });
    });

    // Make header sticky
    header.classList.add('sticky-header');

    // Enable smooth scrolling
    document.documentElement.style.scrollBehavior = 'smooth';

    // Initialize algorithm parameters visibility
    updateAlgorithmParams();
}

// Update visible parameters based on selected algorithm
function updateAlgorithmParams() {
    const algo = algoSelect.value;

    document.querySelectorAll('#lr-params, #k-params, #depth-params, #eps-params, #min-samples-params, #n-components-params, #classification-params').forEach(el => {
        el.classList.add('hidden');
    });

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

document.addEventListener('DOMContentLoaded', () => {
    initUI();

    document.querySelectorAll('input[type="range"], input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            const param = this.id;
            updateParameter(param);
        });
    });

    // Keyboard navigation for buttons
    document.querySelectorAll('button, label[for="upload"]').forEach(button => {
        button.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            }
        });
    });
});

// Add CSS for enhancements
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
    
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .collapse-toggle {
        background: none;
        border: none;
        color: var(--text);
        cursor: pointer;
        margin-left: 0.5rem;
        padding: 0.2rem;
        font-size: 1rem;
        transition: transform 0.3s ease;
    }
    
    .collapsed {
        display: none;
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