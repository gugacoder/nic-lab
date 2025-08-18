/**
 * NIC Lab - JavaScript Principal
 * Funcionalidades globais da aplicação
 */

// Utilitários
const Utils = {
    // Formatação de datas
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('pt-BR', {
            year: 'numeric',
            month: '2-digit', 
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    // Formatação de duração em segundos
    formatDuration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    },

    // Debounce para requisições
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Gerenciamento de notificações
const Notifications = {
    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">×</button>
        `;
        
        // CSS inline para notificações
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: white;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 1rem;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
        `;
        
        // Cores por tipo
        const colors = {
            info: '#5fbcd3',
            success: '#7db46c', 
            warning: '#f39c12',
            error: '#e74c3c'
        };
        notification.style.backgroundColor = colors[type] || colors.info;
        
        document.body.appendChild(notification);
        
        // Auto-remover
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
    }
};

// API Client para comunicação com backend
const ApiClient = {
    async request(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            return await response.text();
            
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    },

    // Métodos específicos da API
    async getPipelineStatus() {
        return await this.request('/api/v1/pipelines/gitlab-qdrant/run');
    },

    async getLastRun() {
        return await this.request('/api/v1/pipelines/gitlab-qdrant/runs/last');
    },

    async runPipeline() {
        return await this.request('/api/v1/pipelines/gitlab-qdrant/run?action=run_pipeline');
    }
};

// Monitoramento de status em tempo real
const StatusMonitor = {
    intervalId: null,
    callbacks: new Set(),

    start(interval = 10000) {
        this.stop(); // Para qualquer monitoramento anterior
        
        this.intervalId = setInterval(async () => {
            try {
                const status = await ApiClient.getPipelineStatus();
                this.callbacks.forEach(callback => callback(status));
            } catch (error) {
                console.error('Status monitoring error:', error);
            }
        }, interval);
    },

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    },

    addCallback(callback) {
        this.callbacks.add(callback);
    },

    removeCallback(callback) {
        this.callbacks.delete(callback);
    }
};

// Theme Management
const ThemeManager = {
    init() {
        this.themeToggle = document.getElementById('theme-toggle');
        this.themeIcon = document.getElementById('theme-icon');
        
        if (!this.themeToggle || !this.themeIcon) return;
        
        // Get saved theme from localStorage or default to 'dark'
        const savedTheme = localStorage.getItem('nic-theme') || 'dark';
        
        // Apply saved theme
        this.applyTheme(savedTheme);
        
        // Add click event listener
        this.themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            this.applyTheme(newTheme);
            localStorage.setItem('nic-theme', newTheme);
        });
    },

    applyTheme(theme) {
        // Set data-theme attribute on html element
        document.documentElement.setAttribute('data-theme', theme);
        
        // Update icon based on theme
        if (this.themeIcon) {
            if (theme === 'light') {
                this.themeIcon.textContent = '☀️'; // Sun icon for light theme
                this.themeToggle.setAttribute('aria-label', 'Mudar para tema escuro');
            } else {
                this.themeIcon.textContent = '🌙'; // Moon icon for dark theme
                this.themeToggle.setAttribute('aria-label', 'Mudar para tema claro');
            }
        }
        
        // Update logo source based on theme (if different logos exist)
        this.updateLogo(theme);
    },

    updateLogo(theme) {
        const logo = document.querySelector('.logo img');
        
        if (!logo) return;
        
        // Check if light theme logo exists, otherwise keep current logo
        if (theme === 'light') {
            // Try to use light logo if available
            const lightLogoPath = '/assets/images/niclab-light-no-margin.svg';
            
            // Test if light logo exists by creating a temporary image
            const testImage = new Image();
            testImage.onload = function() {
                logo.src = lightLogoPath;
            };
            testImage.onerror = function() {
                // Keep existing dark logo if light version doesn't exist
                console.log('Light logo not found, keeping current logo');
            };
            testImage.src = lightLogoPath;
        } else {
            // Use dark logo
            logo.src = '/assets/images/niclab-dark-no-margin.svg';
        }
    },

    getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
};

// Inicialização quando DOM estiver pronto
document.addEventListener('DOMContentLoaded', () => {
    // Adicionar animações CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .notification button {
            background: none;
            border: none;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    `;
    document.head.appendChild(style);

    // Inicializar funcionalidades globais
    ThemeManager.init();
    console.log('🧠 NIC Lab initialized');
});

// Função global para executar pipeline (usada em formulários)
window.runPipeline = async function() {
    try {
        Notifications.show('🚀 Iniciando pipeline...', 'info');
        const result = await ApiClient.runPipeline();
        
        if (result.status === 'job_running') {
            Notifications.show('✅ Pipeline iniciado com sucesso!', 'success');
        } else {
            Notifications.show(`Status: ${result.status}`, 'warning');
        }
        
        // Refresh da página após 2 segundos para mostrar novo status
        setTimeout(() => window.location.reload(), 2000);
        
    } catch (error) {
        Notifications.show(`❌ Erro: ${error.message}`, 'error');
    }
};

// Exportar para uso global
window.NICLab = {
    Utils,
    Notifications,
    ApiClient,
    StatusMonitor,
    ThemeManager
};