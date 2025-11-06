// Kubernetes Copilot Frontend JavaScript

class K8sCopilot {
    constructor() {
        this.baseURL = window.location.origin;
        this.apiURL = `${this.baseURL}/api`;
        this.chatHistory = [];
        this.isLoading = false;
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadSystemData();
        this.setupTabNavigation();
        await this.loadExampleQueries();
        await this.loadDashboardData();
    }

    setupEventListeners() {
        // Query input and submission
        const queryInput = document.getElementById('query-input');
        const askButton = document.getElementById('ask-button');
        
        queryInput.addEventListener('input', (e) => {
            this.updateCharacterCount(e.target.value);
        });
        
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleQuery();
            }
        });
        
        askButton.addEventListener('click', () => this.handleQuery());
        
        // Example query clicks
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('example-query')) {
                queryInput.value = e.target.textContent.trim();
                this.updateCharacterCount(queryInput.value);
                queryInput.focus();
            }
        });
        
        // History item clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('.history-item')) {
                const historyItem = e.target.closest('.history-item');
                const query = historyItem.dataset.query;
                queryInput.value = query;
                this.updateCharacterCount(query);
                queryInput.focus();
            }
        });
    }

    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active tab content
                tabContents.forEach(content => content.classList.remove('active'));
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                // Load specific tab data if needed
                if (tabName === 'costs') {
                    this.loadCostCharts();
                } else if (tabName === 'resources') {
                    this.loadResourceMetrics();
                }
            });
        });
    }

    updateCharacterCount(text) {
        const charCount = document.getElementById('char-count');
        charCount.textContent = text.length;
        
        if (text.length > 450) {
            charCount.style.color = '#dc3545';
        } else if (text.length > 350) {
            charCount.style.color = '#ffc107';
        } else {
            charCount.style.color = '#666';
        }
    }

    async loadSystemData() {
        try {
            const response = await fetch(`${this.apiURL}/stats`);
            if (!response.ok) throw new Error('Failed to load system stats');
            
            const data = await response.json();
            this.updateSystemStatus(data);
        } catch (error) {
            console.error('Error loading system data:', error);
            this.updateSystemStatus(null, error.message);
        }
    }

    updateSystemStatus(data, error = null) {
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const totalDocs = document.getElementById('total-docs');
        
        if (error) {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Error loading system';
            totalDocs.textContent = 'Error';
        } else if (data && data.initialized) {
            statusDot.className = 'status-dot';
            statusText.textContent = 'System Ready';
            totalDocs.textContent = data.total_documents;
        } else {
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'Initializing...';
            totalDocs.textContent = '-';
        }
    }

    async loadExampleQueries() {
        try {
            const response = await fetch(`${this.apiURL}/example-queries`);
            if (!response.ok) throw new Error('Failed to load example queries');
            
            const data = await response.json();
            this.renderExampleQueries(data.queries);
        } catch (error) {
            console.error('Error loading example queries:', error);
            document.getElementById('example-queries').innerHTML = 
                '<div class="error-message">Failed to load examples</div>';
        }
    }

    renderExampleQueries(queries) {
        const container = document.getElementById('example-queries');
        container.innerHTML = queries.map(query => 
            `<div class="example-query">${query}</div>`
        ).join('');
    }

    async handleQuery() {
        if (this.isLoading) return;
        
        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        
        if (!query) {
            this.showError('Please enter a question before clicking Ask!');
            return;
        }
        
        const agentType = document.querySelector('input[name="agent"]:checked').value;
        
        this.isLoading = true;
        this.updateLoadingState(true);
        this.showLoadingResponse(query, agentType);
        
        try {
            const response = await fetch(`${this.apiURL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, agent_type: agentType })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.handleQueryResponse(data);
            
            // Add to history
            this.chatHistory.unshift({
                query: data.query,
                response: data.response,
                agent_type: data.agent_type,
                timestamp: new Date(),
                success: data.success
            });
            
            this.updateChatHistory();
            
        } catch (error) {
            console.error('Query error:', error);
            this.handleQueryResponse({
                query,
                response: '',
                agent_type: agentType,
                success: false,
                error: error.message
            });
        } finally {
            this.isLoading = false;
            this.updateLoadingState(false);
        }
    }

    showLoadingResponse(query, agentType) {
        const responseSection = document.getElementById('response-section');
        responseSection.innerHTML = `
            <div class="loading-animation">
                <div class="loading-spinner"></div>
                <div>Processing your query...</div>
            </div>
            <div class="response-card">
                <div class="response-header">
                    <div class="response-query">${this.escapeHtml(query)}</div>
                    <div class="response-agent">${agentType === 'rag' ? 'RAG Agent' : 'Copilot Agent'}</div>
                </div>
                <div class="response-content">Thinking...</div>
            </div>
        `;
    }

    handleQueryResponse(data) {
        const responseSection = document.getElementById('response-section');
        
        if (data.success) {
            responseSection.innerHTML = `
                <div class="response-card">
                    <div class="response-header">
                        <div class="response-query">${this.escapeHtml(data.query)}</div>
                        <div class="response-agent">${data.agent_type}</div>
                    </div>
                    <div class="response-content">${this.escapeHtml(data.response)}</div>
                </div>
            `;
        } else {
            responseSection.innerHTML = `
                <div class="response-card error">
                    <div class="response-header">
                        <div class="response-query">${this.escapeHtml(data.query)}</div>
                        <div class="response-agent">${data.agent_type}</div>
                    </div>
                    <div class="response-content">
                        <div class="error-message">
                            <strong>Error:</strong> ${this.escapeHtml(data.error || 'Unknown error occurred')}
                        </div>
                    </div>
                </div>
            `;
        }
    }

    updateChatHistory() {
        const historyList = document.getElementById('history-list');
        
        if (this.chatHistory.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No previous conversations</p>';
            return;
        }
        
        const recentHistory = this.chatHistory.slice(1, 6); // Skip the most recent (shown above)
        
        if (recentHistory.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No previous conversations</p>';
            return;
        }
        
        historyList.innerHTML = recentHistory.map(item => `
            <div class="history-item" data-query="${this.escapeHtml(item.query)}">
                <div class="history-query">${this.escapeHtml(item.query)}</div>
                <div class="history-preview">${this.escapeHtml(item.response.substring(0, 100))}${item.response.length > 100 ? '...' : ''}</div>
            </div>
        `).join('');
    }

    updateLoadingState(loading) {
        const askButton = document.getElementById('ask-button');
        askButton.disabled = loading;
        askButton.textContent = loading ? 'Processing...' : 'Ask';
    }

    async loadDashboardData() {
        try {
            const response = await fetch(`${this.apiURL}/cost-data`);
            if (!response.ok) throw new Error('Failed to load dashboard data');
            
            const data = await response.json();
            if (data.success) {
                this.dashboardData = data.data;
                this.updateCostMetrics();
                this.updateResourceMetrics();
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    updateCostMetrics() {
        if (!this.dashboardData || !this.dashboardData.deployment_costs) {
            document.getElementById('total-cost').textContent = 'No data';
            document.getElementById('expensive-deployment').textContent = 'No data';
            document.getElementById('potential-savings').textContent = 'No data';
            return;
        }
        
        const costs = this.dashboardData.deployment_costs;
        const totalCost = costs.reduce((sum, item) => sum + item.total_cost, 0);
        
        // Find most expensive deployment
        const expensiveDeployment = costs.reduce((max, item) => 
            item.total_cost > max.total_cost ? item : max
        );
        
        document.getElementById('total-cost').textContent = `$${totalCost.toFixed(2)}`;
        document.getElementById('expensive-deployment').textContent = expensiveDeployment.deployment;
        document.getElementById('potential-savings').textContent = `$${(totalCost * 0.2).toFixed(2)}`;
    }

    updateResourceMetrics() {
        // Mock resource data since it's not in the cost data
        const resourceMetrics = {
            'cpu-util': '65%',
            'memory-util': '78%',
            'storage-util': '42%',
            'network-util': '23%',
            'total-nodes': '5',
            'total-pods': '23',
            'total-deployments': '8',
            'total-gpus': '4'
        };
        
        Object.entries(resourceMetrics).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    loadCostCharts() {
        if (!this.dashboardData || !this.dashboardData.deployment_costs) return;
        
        const costs = this.dashboardData.deployment_costs;
        
        // Group costs by deployment
        const deploymentTotals = {};
        costs.forEach(item => {
            if (!deploymentTotals[item.deployment]) {
                deploymentTotals[item.deployment] = 0;
            }
            deploymentTotals[item.deployment] += item.total_cost;
        });
        
        // Cost chart
        const costChart = document.getElementById('cost-chart');
        if (costChart) {
            new Chart(costChart, {
                type: 'bar',
                data: {
                    labels: Object.keys(deploymentTotals),
                    datasets: [{
                        label: 'Total Cost ($)',
                        data: Object.values(deploymentTotals),
                        backgroundColor: 'rgba(102, 126, 234, 0.6)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Total Cost by Deployment (30 days)'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Breakdown chart (pie chart of cost types)
        const breakdownChart = document.getElementById('breakdown-chart');
        if (breakdownChart) {
            const costTypes = {
                CPU: costs.reduce((sum, item) => sum + item.cpu_cost, 0),
                Memory: costs.reduce((sum, item) => sum + item.memory_cost, 0),
                Storage: costs.reduce((sum, item) => sum + item.storage_cost, 0),
                Network: costs.reduce((sum, item) => sum + item.network_cost, 0)
            };
            
            new Chart(breakdownChart, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(costTypes),
                    datasets: [{
                        data: Object.values(costTypes),
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.8)',
                            'rgba(118, 75, 162, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(40, 167, 69, 0.8)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Cost Breakdown by Type'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }

    loadResourceMetrics() {
        // This would typically load real resource data
        // For now, we'll just ensure the metrics are displayed
        this.updateResourceMetrics();
    }

    showError(message) {
        const responseSection = document.getElementById('response-section');
        responseSection.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${this.escapeHtml(message)}
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new K8sCopilot();
});
