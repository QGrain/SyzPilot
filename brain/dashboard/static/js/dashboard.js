// Global state
let authToken = null;
let autoRefreshInterval = null;
let logRefreshIntervals = {};  // task_id -> interval

// Check authentication on page load
document.addEventListener('DOMContentLoaded', () => {
    authToken = localStorage.getItem('dashboardToken');

    if (!authToken) {
        window.location.href = '/login';
        return;
    }

    initDashboard();
});

async function initDashboard() {
    try {
        const isValid = await verifyToken();
        if (!isValid) {
            logout();
            return;
        }

        await refreshData();

        document.getElementById('loadingState').classList.add('d-none');
        document.getElementById('mainContent').classList.remove('d-none');

        startAutoRefresh();

    } catch (error) {
        console.error('Dashboard initialization failed:', error);
        showToast('Failed to initialize dashboard', 'danger');
    }
}

async function verifyToken() {
    try {
        const response = await fetch('/api/verify_token', {
            method: 'POST',
            headers: {'Authorization': `Bearer ${authToken}`}
        });

        if (!response.ok) return false;
        const data = await response.json();
        return data.valid === true;
    } catch (error) {
        return false;
    }
}

async function refreshData() {
    try {
        const [statsResponse, tasksResponse] = await Promise.all([
            apiRequest('/api/stats'),
            apiRequest('/list_tasks')
        ]);

        updateStats(statsResponse);
        updateTasksView(tasksResponse.list_tasks || []);
        updateConnectionStatus(true);
    } catch (error) {
        console.error('Failed to refresh data:', error);
        updateConnectionStatus(false);
        showToast('Failed to refresh data', 'danger');
    }
}

function updateStats(stats) {
    document.getElementById('statActiveTasks').textContent = stats.active_tasks || 0;
    document.getElementById('statTotalRuns').textContent = stats.total_runs || 0;
    document.getElementById('statAvailablePorts').textContent =
        (stats.available_grpc_ports || 0) + (stats.available_tunnel_ports || 0);

    const torchServeBadge = document.getElementById('statTorchServe');
    if (stats.torchserve_status === 'running') {
        torchServeBadge.textContent = 'Running';
        torchServeBadge.className = 'badge bg-success';
    } else {
        torchServeBadge.textContent = 'Stopped';
        torchServeBadge.className = 'badge bg-secondary';
    }
}

function updateTasksView(tasks) {
    const container = document.getElementById('tasksContainer');
    const noTasksMsg = document.getElementById('noTasksMessage');

    if (tasks.length === 0) {
        noTasksMsg.classList.remove('d-none');
        container.classList.add('d-none');
        return;
    }

    noTasksMsg.classList.add('d-none');
    container.classList.remove('d-none');

    container.innerHTML = tasks.map(task => createTaskCard(task)).join('');

    // Restore expanded state after rendering
    tasks.forEach(task => {
        const isExpanded = sessionStorage.getItem(`task-${task.task_id}-expanded`);
        if (isExpanded === 'true') {
            // Need to use setTimeout to ensure DOM is ready
            setTimeout(() => toggleTaskDetails(task.task_id), 0);
        }
    });
}

function createTaskCard(task) {
    const cardId = `task-${task.task_id.replace(/[^a-zA-Z0-9]/g, '_')}`;

    return `
        <div class="card mb-3 task-card">
            <div class="card-header" style="cursor: pointer;" onclick="toggleTaskDetails('${escapeHtml(task.task_id)}')">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-0">
                            <i class="bi bi-chevron-right" id="chevron-${escapeHtml(task.task_id)}"></i>
                            <strong>${escapeHtml(task.task_name)}</strong>
                            <span class="badge bg-secondary">Run ${task.run_id}</span>
                            <span class="badge mode-${task.mode}">${task.mode}</span>
                        </h6>
                        <small class="text-muted">Task ID: <code>${escapeHtml(task.task_id)}</code></small>
                    </div>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="event.stopPropagation(); pingTask('${escapeHtml(task.task_id)}')" title="Ping">
                            <i class="bi bi-send"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="event.stopPropagation(); unregisterTask('${escapeHtml(task.task_id)}')" title="Unregister">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
            <div class="collapse" id="details-${escapeHtml(task.task_id)}">
                <div class="card-body">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <p><strong>Fuzzer ID:</strong> <code>${escapeHtml(task.fuzzer_id)}</code></p>
                            <p><strong>gRPC Port:</strong> ${task.grpc_port}</p>
                            <p><strong>Tunnel Port:</strong> ${task.tunnel_port || 'N/A'}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Callback Addr:</strong> ${task.callback_addr || 'N/A'}</p>
                            <p><strong>Receiver PID:</strong> ${task.receiver_pid || 'N/A'}</p>
                            <p><strong>Trainer PID:</strong> ${task.trainer_pid || 'N/A'}</p>
                            <p><strong>Attributor PID:</strong> ${task.attributor_pid || 'N/A'}</p>
                            <p><strong>Registered:</strong> ${formatTimestamp(task.registered_at)}</p>
                        </div>
                    </div>

                    <!-- Process Logs Tabs -->
                    <ul class="nav nav-tabs" role="tablist">
                        <li class="nav-item">
                            <a class="nav-link active" data-bs-toggle="tab" href="#receiver-${escapeHtml(task.task_id)}" role="tab">
                                <i class="bi bi-diagram-3"></i> Receiver
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" data-bs-toggle="tab" href="#trainer-${escapeHtml(task.task_id)}" role="tab">
                                <i class="bi bi-gear"></i> Trainer
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" data-bs-toggle="tab" href="#attributor-${escapeHtml(task.task_id)}" role="tab">
                                <i class="bi bi-graph-up"></i> Attributor
                            </a>
                        </li>
                    </ul>

                    <div class="tab-content border border-top-0 p-3" style="background: #f8f9fa;">
                        <div class="tab-pane fade show active" id="receiver-${escapeHtml(task.task_id)}" role="tabpanel">
                            <div class="log-viewer" id="log-receiver-${escapeHtml(task.task_id)}">
                                <div class="text-muted">Loading logs...</div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="trainer-${escapeHtml(task.task_id)}" role="tabpanel">
                            <div class="log-viewer" id="log-trainer-${escapeHtml(task.task_id)}">
                                <div class="text-muted">Loading logs...</div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="attributor-${escapeHtml(task.task_id)}" role="tabpanel">
                            <div class="log-viewer" id="log-attributor-${escapeHtml(task.task_id)}">
                                <div class="text-muted">Loading logs...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function toggleTaskDetails(taskId) {
    const detailsEl = document.getElementById(`details-${taskId}`);
    const chevronEl = document.getElementById(`chevron-${taskId}`);

    if (detailsEl.classList.contains('show')) {
        // Collapse
        detailsEl.classList.remove('show');
        chevronEl.classList.remove('bi-chevron-down');
        chevronEl.classList.add('bi-chevron-right');

        // Stop log refresh
        if (logRefreshIntervals[taskId]) {
            clearInterval(logRefreshIntervals[taskId]);
            delete logRefreshIntervals[taskId];
        }
        sessionStorage.removeItem(`task-${taskId}-expanded`);
    } else {
        // Expand
        detailsEl.classList.add('show');
        chevronEl.classList.remove('bi-chevron-right');
        chevronEl.classList.add('bi-chevron-down');

        // Load initial logs
        await refreshTaskLogs(taskId, 'receiver');
        await refreshTaskLogs(taskId, 'trainer');
        await refreshTaskLogs(taskId, 'attributor');

        // Start auto-refresh for logs
        logRefreshIntervals[taskId] = setInterval(async () => {
            const activeTab = document.querySelector(`#details-${taskId} .nav-link.active`);
            if (activeTab) {
                const processType = activeTab.getAttribute('href').split('-')[0].substring(1);
                await refreshTaskLogs(taskId, processType);
            }
        }, 3000);  // Refresh every 3 seconds
        sessionStorage.setItem(`task-${taskId}-expanded`, 'true');
    }
}

async function refreshTaskLogs(taskId, processType) {
    try {
        const response = await apiRequest(`/api/task/${taskId}/logs/${processType}?lines=50`);
        const logEl = document.getElementById(`log-${processType}-${taskId}`);

        if (!logEl) return;

        if (response.logs.length === 0) {
            logEl.innerHTML = '<div class="text-muted">No logs yet</div>';
        } else {
            logEl.innerHTML = response.logs.map(line =>
                `<div class="log-line">${escapeHtml(line)}</div>`
            ).join('');

            // Auto-scroll to bottom
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (error) {
        console.error(`Failed to load ${processType} logs:`, error);
    }
}

async function pingTask(taskId) {
    try {
        const response = await apiRequest(`/ping_fuzzer/${taskId}`);
        showToast(`Ping successful: ${response.response}`, 'success');
    } catch (error) {
        showToast(`Ping failed: ${error.message}`, 'danger');
    }
}

async function unregisterTask(taskId) {
    if (!confirm(`Are you sure you want to unregister task ${taskId}?`)) {
        return;
    }

    try {
        await apiRequest(`/unregister/${taskId}`, { method: 'POST' });
        showToast('Task unregistered successfully', 'success');
        await refreshData();
    } catch (error) {
        showToast(`Failed to unregister task: ${error.message}`, 'danger');
    }
}

// Helper functions
async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
            ...options.headers
        }
    });

    if (!response.ok) {
        if (response.status === 401) {
            logout();
            throw new Error('Unauthorized');
        }
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Request failed');
    }

    return response.json();
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    const toastId = `toast-${Date.now()}`;

    const bgClass = type === 'success' ? 'bg-success' :
                    type === 'danger' ? 'bg-danger' :
                    type === 'warning' ? 'bg-warning' : 'bg-info';

    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white ${bgClass} border-0`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${escapeHtml(message)}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toast);

    const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
    bsToast.show();

    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connectionStatus');
    if (connected) {
        statusEl.innerHTML = '<i class="bi bi-circle-fill"></i> Connected';
        statusEl.className = 'badge bg-success me-3';
    } else {
        statusEl.innerHTML = '<i class="bi bi-circle-fill"></i> Disconnected';
        statusEl.className = 'badge bg-danger me-3';
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    autoRefreshInterval = setInterval(refreshData, 600000);  // 600 seconds
}

function logout() {
    localStorage.removeItem('dashboardToken');
    window.location.href = '/login';
}