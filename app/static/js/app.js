/**
 * AlyaAloft - Main JavaScript functionality
 * Handles chat interface, document management, WebSocket connection
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // App state
    const appState = {
        websocket: null,
        websocketReconnectTimer: null,
        selectedDocument: null,
        documents: [],
        connecting: false,
        connected: false,
    };
    
    // DOM Elements containers
    let connectionStatus, mobileConnectionStatus, uploadForm, uploadButton;
    let uploadProgress, uploadProgressBar, uploadStatus, documentsList;
    let chatTitle, chatSubtitle, messagesContainer, chatForm;
    let messageInput, sendButton, typingIndicator;
    let messageTemplate, documentTemplate;
    
    // Initialize the app
    initialize();
    
    /**
     * Get DOM elements and store references
     */
    function getElements() {
        // Connection elements
        connectionStatus = document.getElementById('connection-status');
        mobileConnectionStatus = document.getElementById('mobile-connection-status');
        
        // Upload form elements
        uploadForm = document.getElementById('upload-form');
        uploadButton = document.getElementById('upload-button');
        uploadProgress = document.getElementById('upload-progress');
        uploadProgressBar = document.getElementById('upload-progress-bar');
        uploadStatus = document.getElementById('upload-status');
        
        // Document elements
        documentsList = document.getElementById('documents-list');
        
        // Chat elements
        chatTitle = document.getElementById('chat-title');
        chatSubtitle = document.getElementById('chat-subtitle');
        messagesContainer = document.getElementById('messages-container');
        chatForm = document.getElementById('chat-form');
        messageInput = document.getElementById('message-input');
        sendButton = document.getElementById('send-button');
        typingIndicator = document.getElementById('typing-indicator');
        
        // Templates
        messageTemplate = document.getElementById('message-template');
        documentTemplate = document.getElementById('document-template');
    }
    
    /**
     * Initialize the application
     */
    function initialize() {
        // Get DOM elements
        getElements();
        
        // Set up event listeners
        setupEventListeners();
        
        // Set up WebSocket
        initializeWebSocket();
        
        // Load documents
        loadDocuments();
        
        // Set up document status polling
        setupDocumentPolling();
        
        // Show document selection notification if in chat interface
        if (window.location.pathname.includes('/chat')) {
            showDocumentSelectionNotification();
        }
    }
    
    /**
     * Initialize WebSocket connection
     */
    function initializeWebSocket() {
        // Clear any reconnect timer
        if (appState.websocketReconnectTimer) {
            clearTimeout(appState.websocketReconnectTimer);
            appState.websocketReconnectTimer = null;
        }
        
        // Close existing connection if any
        if (appState.websocket) {
            try {
                if (appState.websocket.readyState !== WebSocket.CLOSED) {
                    appState.websocket.close();
                }
            } catch (e) {
                console.error('Error closing existing WebSocket:', e);
            }
            appState.websocket = null;
        }
        
        // Update connection status
        appState.connecting = true;
        appState.connected = false;
        updateConnectionStatus('connecting');
        
        try {
            // Create new WebSocket connection with absolute URL to avoid path issues
            // This ensures we're connecting to the right WebSocket endpoint
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v1/chat/ws`;
            
            console.log('Connecting to WebSocket at:', wsUrl);
            
            appState.websocket = new WebSocket(wsUrl);
            
            // Add a connection timeout
            const connectionTimeout = setTimeout(() => {
                if (appState.connecting && !appState.connected) {
                    console.error('WebSocket connection timeout');
                    appState.connecting = false;
                    updateConnectionStatus('disconnected');
                    
                    // Try to reconnect after delay
                    scheduleReconnect();
                }
            }, 5000);
            
            // Set up WebSocket event handlers
            appState.websocket.onopen = (event) => {
                clearTimeout(connectionTimeout);
                handleWebSocketOpen(event);
            };
            
            appState.websocket.onmessage = handleWebSocketMessage;
            
            appState.websocket.onclose = (event) => {
                clearTimeout(connectionTimeout);
                handleWebSocketClose(event);
            };
            
            appState.websocket.onerror = (event) => {
                clearTimeout(connectionTimeout);
                handleWebSocketError(event);
            };
            
        } catch (error) {
            console.error('Error initializing WebSocket:', error);
            appState.connecting = false;
            updateConnectionStatus('disconnected');
            
            // Try to reconnect after delay
            scheduleReconnect();
        }
    }
    
    /**
     * Schedule WebSocket reconnection
     */
    function scheduleReconnect() {
        if (!appState.websocketReconnectTimer) {
            console.log('Scheduling WebSocket reconnection in 3 seconds');
            appState.websocketReconnectTimer = setTimeout(() => {
                console.log('Attempting to reconnect WebSocket...');
                initializeWebSocket();
            }, 3000);
        }
    }
    
    /**
     * Handle WebSocket open event
     */
    function handleWebSocketOpen(event) {
        console.log('WebSocket connection established');
        appState.connecting = false;
        appState.connected = true;
        updateConnectionStatus('connected');
        
        // Send a ping every 30 seconds to keep the connection alive
        setInterval(() => {
            if (appState.connected) {
                sendWebSocketMessage({
                    type: 'ping',
                    timestamp: new Date().toISOString()
                });
            }
        }, 30000);
    }
    
    /**
     * Handle WebSocket message event
     * @param {MessageEvent} event - WebSocket message event
     */
    function handleWebSocketMessage(event) {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
            case 'connection_established':
                handleConnectionEstablished(data);
                break;
            
            case 'chat_message':
                handleChatMessage(data.message);
                break;
            
            case 'processing':
                handleProcessingMessage(data);
                break;
            
            case 'error':
                showToast(data.message, 'error');
                typingIndicator.classList.add('hidden');
                break;
            
            case 'pong':
                // Ping response received, connection is alive
                updateConnectionStatus('connected');
                break;
            
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    /**
     * Handle WebSocket close event
     * @param {CloseEvent} event - WebSocket close event
     */
    function handleWebSocketClose(event) {
        console.log('WebSocket connection closed:', event.code, event.reason);
        appState.connecting = false;
        appState.connected = false;
        updateConnectionStatus('disconnected');
        
        // Attempt to reconnect after a delay
        appState.websocketReconnectTimer = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            initializeWebSocket();
        }, 3000);
    }
    
    /**
     * Handle WebSocket error event
     * @param {Event} event - WebSocket error event
     */
    function handleWebSocketError(event) {
        console.error('WebSocket error:', event);
        appState.connecting = false;
        appState.connected = false;
        updateConnectionStatus('disconnected');
    }
    
    /**
     * Update the connection status indicators
     * @param {string} status - Connection status (connecting, connected, disconnected)
     */
    function updateConnectionStatus(status) {
        // Remove all status classes
        if (connectionStatus) {
            connectionStatus.classList.remove('connecting', 'connected', 'disconnected');
            connectionStatus.classList.add(status);
        }
        
        if (mobileConnectionStatus) {
            mobileConnectionStatus.classList.remove('connecting', 'connected', 'disconnected');
            mobileConnectionStatus.classList.add(status);
        }
        
        // Update the text
        let statusText = 'Disconnected';
        if (status === 'connecting') {
            statusText = 'Connecting...';
        } else if (status === 'connected') {
            statusText = 'Connected';
        }
        
        if (connectionStatus) {
            connectionStatus.textContent = statusText;
        }
        
        if (mobileConnectionStatus) {
            mobileConnectionStatus.textContent = statusText;
        }
    }
    
    /**
     * Send a message via WebSocket
     * @param {Object} message - Message to send
     */
    function sendWebSocketMessage(message) {
        if (appState.connected && appState.websocket) {
            appState.websocket.send(JSON.stringify(message));
        } else {
            console.error('Cannot send message, WebSocket is not connected');
            showToast('Connection lost. Reconnecting...', 'error');
            // Attempt to reconnect
            initializeWebSocket();
        }
    }
    
    /**
     * Set up event listeners for the app
     */
    function setupEventListeners() {
        // Document upload form submission
        if (uploadForm) {
            uploadForm.addEventListener('submit', handleDocumentUpload);
        }
        
        // Chat form submission
        if (chatForm) {
            chatForm.addEventListener('submit', handleChatSubmit);
        }
    }
    
    /**
     * Load documents from the server
     */
    async function loadDocuments() {
        try {
            const response = await fetch('/api/v1/documents');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            appState.documents = data || [];
            
            renderDocumentsList();
        } catch (err) {
            console.error('Error loading documents:', err);
            appState.documents = [];
            renderDocumentsList();
        }
    }
    
    /**
     * Render the documents list
     */
    function renderDocumentsList() {
        // Clear the list
        documentsList.innerHTML = '';
        
        if (appState.documents.length === 0) {
            documentsList.innerHTML = `
                <div class="flex items-center justify-center h-32 text-gray-500">
                    <p>No documents yet</p>
                </div>
            `;
            return;
        }
        
        // Add each document
        appState.documents.forEach(doc => {
            const docElement = documentTemplate.content.cloneNode(true);
            const docContainer = docElement.querySelector('.document');
            
            // Set document ID
            docContainer.dataset.documentId = doc.id;
            
            // Set document title
            docElement.querySelector('.document-title').textContent = doc.title || doc.filename;
            
            // Set document status
            const statusElement = docElement.querySelector('.document-status');
            let statusClass = 'pending';
            let statusText = 'Pending';
            
            if (doc.processing_status === 'processing') {
                statusClass = 'processing';
                statusText = 'Processing';
            } else if (doc.processing_status === 'completed' || doc.processed) {
                statusClass = 'completed';
                statusText = 'Ready';
            } else if (doc.processing_status === 'failed') {
                statusClass = 'failed';
                statusText = 'Failed';
            }
            
            statusElement.classList.add(statusClass);
            statusElement.textContent = statusText;
            
            // Set document metadata
            const metaElement = docElement.querySelector('.document-meta');
            const uploadDate = new Date(doc.upload_time);
            
            // Format page count display properly
            let pageCountText = '';
            if (doc.page_count && doc.page_count > 0) {
                pageCountText = `${doc.page_count} pages`;
            } else if (doc.processing_status === 'pending' || doc.processing_status === 'processing') {
                pageCountText = 'Processing...';
            } else {
                pageCountText = 'Unknown pages';
            }
            
            metaElement.textContent = `${pageCountText} · Uploaded ${formatDate(uploadDate)}`;
            
            // Add delete button
            const actionsElement = docElement.querySelector('.document-actions');
            const deleteButton = document.createElement('button');
            deleteButton.className = 'text-red-600 hover:text-red-800 focus:outline-none';
            deleteButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                </svg>
            `;
            // Prevent click event from bubbling to the document container
            deleteButton.addEventListener('click', async (e) => {
                e.stopPropagation();
                await deleteDocument(doc.id);
            });
            actionsElement.appendChild(deleteButton);
            
            // Mark as selected if this is the current document
            if (appState.selectedDocument && doc.id === appState.selectedDocument.id) {
                docContainer.classList.add('selected');
            }
            
            // Add click event
            docContainer.addEventListener('click', () => selectDocument(doc));
            
            // Add to list
            documentsList.appendChild(docElement);
        });
    }
    
    /**
     * Select a document to chat with
     * @param {Object} doc - Document object
     */
    async function selectDocument(doc) {
        // Clear previous selection
        const previousSelected = document.querySelector('.document.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected', 'border-primary-500', 'bg-primary-50');
        }
        
        // Mark as selected
        const docElement = document.querySelector(`.document[data-document-id="${doc.id}"]`);
        if (docElement) {
            docElement.classList.add('selected', 'border-primary-500', 'bg-primary-50');
        }
        
        // Set as current document
        appState.selectedDocument = doc;
        
        // Update chat header
        if (chatTitle) {
            chatTitle.textContent = doc.title || doc.filename || 'Untitled Document';
        }
        
        if (chatSubtitle) {
            const uploadDate = doc.upload_time ? formatDate(new Date(doc.upload_time)) : 'Unknown date';
            const pageCount = doc.page_count || 0;
            chatSubtitle.textContent = `Uploaded ${uploadDate} • ${pageCount} pages`;
        }
        
        // Enable message input
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.placeholder = "Ask a question about the document...";
        }
        
        if (sendButton) {
            sendButton.disabled = false;
        }
        
        // Hide document selection notification
        hideDocumentSelectionNotification();
        
        // Show chat interface
        if (messagesContainer) {
            // Clear any placeholder content
            messagesContainer.innerHTML = '';
            
            // Add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'flex items-center justify-center h-32 text-gray-500';
            loadingIndicator.innerHTML = `
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-primary-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Loading chat history...</span>
            `;
            messagesContainer.appendChild(loadingIndicator);
        }
        
        // Load chat history
        await loadChatHistory(doc.id);
    }
    
    /**
     * Load chat history for a document
     * @param {string} docId - Document ID
     */
    async function loadChatHistory(docId) {
        try {
            // Clear messages
            messagesContainer.innerHTML = '';
            
            const response = await fetch(`/api/v1/documents/${docId}/chat`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            if (!data.messages || data.messages.length === 0) {
                messagesContainer.innerHTML = `
                    <div class="flex items-center justify-center h-full text-gray-500">
                        <div class="text-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-gray-300 mb-4" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd" />
                            </svg>
                            <p class="text-lg font-medium">No messages yet</p>
                            <p class="mt-1">Start a conversation by asking a question</p>
                        </div>
                    </div>
                `;
                return;
            }
            
            // Render messages
            const fragment = document.createDocumentFragment();
            data.messages.forEach(message => {
                const messageElement = createMessageElement(message);
                fragment.appendChild(messageElement);
            });
            
            messagesContainer.innerHTML = '';
            messagesContainer.appendChild(fragment);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        } catch (err) {
            console.error('Error loading chat history:', err);
            showToast('Failed to load chat history', 'error');
        }
    }
    
    /**
     * Handle document upload form submission
     * @param {Event} event - Form submission event
     */
    async function handleDocumentUpload(event) {
        event.preventDefault();
        
        const titleInput = document.getElementById('title');
        const fileInput = document.getElementById('file');
        
        if (!fileInput.files.length) {
            showToast('Please select a file to upload', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Check file type
        if (!file.type.includes('pdf')) {
            showToast('Please upload a PDF file', 'error');
            return;
        }
        
        // Check file size (max 10MB by default)
        const maxSizeMB = 10; // This should match server-side limit
        const maxSizeBytes = maxSizeMB * 1024 * 1024;
        
        if (file.size > maxSizeBytes) {
            showToast(`File size exceeds maximum of ${maxSizeMB}MB`, 'error');
            return;
        }
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        if (titleInput.value.trim()) {
            formData.append('title', titleInput.value.trim());
        }
        
        // Show progress
        const uploadProgress = document.getElementById('upload-progress');
        const uploadProgressBar = document.getElementById('upload-progress-bar');
        const uploadStatus = document.getElementById('upload-status');
        const uploadButton = document.getElementById('upload-button');
        
        uploadProgress.classList.remove('hidden');
        uploadButton.disabled = true;
        uploadStatus.textContent = 'Uploading...';
        
        try {
            // Upload file
            const response = await fetch('/api/v1/documents/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update progress to processing
            uploadProgressBar.style.width = '100%';
            uploadStatus.textContent = 'Processing document...';
            
            // Reset form
            uploadForm.reset();
            document.getElementById('file-name').textContent = 'Select PDF file';
            
            // Add to documents list
            appState.documents.push(data);
            renderDocumentsList();
            
            // Select the uploaded document and hide notification
            selectDocument(data);
            hideDocumentSelectionNotification();
            
            // Show success message and reset progress
            showToast('Document uploaded successfully!', 'success');
            
            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                uploadButton.disabled = false;
                uploadProgressBar.style.width = '0%';
            }, 1000);
            
        } catch (err) {
            console.error('Error uploading document:', err);
            showToast('Error uploading document. Please try again.', 'error');
            
            uploadProgress.classList.add('hidden');
            uploadButton.disabled = false;
        }
    }
    
    /**
     * Handle chat form submission
     * @param {Event} event - Form submit event
     */
    function handleChatSubmit(event) {
        event.preventDefault();
        
        if (!appState.selectedDocument) {
            showToast('Please select a document first', 'error');
            return;
        }
        
        const message = messageInput.value.trim();
        if (!message) {
            return;
        }
        
        // Create message object
        const messageObj = {
            type: 'chat_message',
            document_id: appState.selectedDocument.id,
            content: message,
            timestamp: new Date().toISOString()
        };
        
        // Send via WebSocket
        sendWebSocketMessage(messageObj);
        
        // Add user message to UI
        const userMessage = {
            id: `temp-${Date.now()}`,
            type: 'user',
            content: message,
            timestamp: new Date().toISOString()
        };
        
        addMessageToUI(userMessage);
        
        // Clear input
        messageInput.value = '';
        
        // Show typing indicator
        typingIndicator.classList.remove('hidden');
    }
    
    /**
     * Handle incoming chat message
     * @param {Object} message - Message object
     */
    function handleChatMessage(message) {
        // Hide typing indicator
        typingIndicator.classList.add('hidden');
        
        // Remove any temporary processing indicator
        const processingIndicator = document.getElementById('processing-indicator');
        if (processingIndicator) {
            processingIndicator.remove();
        }
        
        // Add message to UI
        addMessageToUI(message);
    }
    
    /**
     * Add a message to the UI
     * @param {Object} message - Message object
     */
    function addMessageToUI(message) {
        const messageElement = createMessageElement(message);
        
        // Add to container
        messagesContainer.appendChild(messageElement);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    /**
     * Create a message element
     * @param {Object} message - Message object
     * @returns {HTMLElement} Message element
     */
    function createMessageElement(message) {
        const messageElement = messageTemplate.content.cloneNode(true);
        const messageContainer = messageElement.querySelector('.message');
        const messageIcon = messageElement.querySelector('.message-icon');
        const messageText = messageElement.querySelector('.message-text');
        const messageMeta = messageElement.querySelector('.message-meta');
        
        // Add appropriate classes
        messageContainer.classList.add(message.type);
        messageIcon.classList.add(message.type);
        
        // Set icon
        if (message.type === 'user') {
            messageIcon.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                </svg>
            `;
        } else {
            // System message - check if advanced model was used
            const usedAdvancedModel = message.model_type === 'llama2';
            messageIcon.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M14.243 5.757a6 6 0 10-.986 9.284 1 1 0 111.087 1.678A8 8 0 1118 10a3 3 0 01-4.8 2.401A4 4 0 1114 10a1 1 0 102 0c0-1.537-.586-3.07-1.757-4.243zM12 10a2 2 0 10-4 0 2 2 0 004 0z" clip-rule="evenodd" />
                </svg>
            `;
            
            // Add model badge if advanced model was used
            if (usedAdvancedModel) {
                const modelBadge = document.createElement('div');
                modelBadge.className = 'absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border border-white';
                modelBadge.title = 'Generated using advanced Llama 2 model';
                messageIcon.style.position = 'relative';
                messageIcon.appendChild(modelBadge);
            }
        }
        
        // Set message text (support simple markdown for code blocks)
        messageText.innerHTML = formatMessageContent(message.content);
        
        // Set metadata
        const messageTime = new Date(message.timestamp);
        messageMeta.textContent = formatTime(messageTime);
        
        return messageContainer;
    }
    
    /**
     * Format message content with simple markdown support
     * @param {string} content - Message content
     * @returns {string} Formatted HTML
     */
    function formatMessageContent(content) {
        if (!content) return '';
        
        // Escape HTML first
        let html = escapeHtml(content);
        
        // Convert code blocks (```code```)
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Convert inline code (`code`)
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert links
        html = html.replace(
            /https?:\/\/[^\s)]+/g, 
            '<a href="$&" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline">$&</a>'
        );
        
        // Convert newlines to <br>
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }
    
    /**
     * Escape HTML special characters
     * @param {string} text - Text to escape
     * @returns {string} Escaped HTML
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Format date for display
     * @param {Date} date - Date to format
     * @returns {string} Formatted date
     */
    function formatDate(date) {
        if (!(date instanceof Date)) {
            return 'Unknown date';
        }
        
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        
        const isToday = date.toDateString() === today.toDateString();
        const isYesterday = date.toDateString() === yesterday.toDateString();
        
        if (isToday) {
            return `Today at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        } else if (isYesterday) {
            return `Yesterday at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        } else {
            return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
        }
    }
    
    /**
     * Format time for display
     * @param {Date} date - Date to format
     * @returns {string} Formatted time
     */
    function formatTime(date) {
        if (!(date instanceof Date)) {
            return 'Unknown time';
        }
        
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    /**
     * Show a toast notification
     * @param {string} message - Notification message
     * @param {string} type - Notification type (success, error, info)
     */
    function showToast(message, type = 'info') {
        // Check if toast container exists, create if not
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container fixed bottom-4 right-4 z-50 flex flex-col-reverse items-end space-y-reverse space-y-2';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = 'toast flex items-center p-4 rounded-lg shadow-lg max-w-xs transform transition-all duration-300 ease-in-out translate-y-0 opacity-100 slide-up';
        
        // Set background color based on type
        if (type === 'success') {
            toast.classList.add('bg-green-600', 'text-white');
        } else if (type === 'error') {
            toast.classList.add('bg-red-600', 'text-white');
        } else {
            toast.classList.add('bg-blue-600', 'text-white');
        }
        
        // Add icon based on type
        let icon = '';
        if (type === 'success') {
            icon = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
            `;
        } else if (type === 'error') {
            icon = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                </svg>
            `;
        } else {
            icon = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zm-1 8a1 1 0 01-1-1v-3a1 1 0 112 0v3a1 1 0 01-1 1z" clip-rule="evenodd" />
                </svg>
            `;
        }
        
        // Set content
        toast.innerHTML = `
            ${icon}
            <span>${message}</span>
        `;
        
        // Add to container
        toastContainer.prepend(toast);
        
        // Remove after delay
        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-2');
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 5000);
    }
    
    /**
     * Handle processing message
     * @param {Object} data - Processing message data
     */
    function handleProcessingMessage(data) {
        // Show improved typing indicator with custom message
        typingIndicator.classList.remove('hidden');
        
        // Check if there's already a processing indicator and remove it
        const existingIndicator = document.getElementById('processing-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        // Create a processing message element properly
        const processingMessage = {
            id: `processing-${Date.now()}`,
            type: 'system',
            content: 'Analyzing document...',
            timestamp: data.timestamp,
            isProcessing: true
        };
        
        // Create the message element
        const messageElement = createMessageElement(processingMessage);
        messageElement.id = 'processing-indicator';
        
        // Get the message-text element within the message element
        const messageTextEl = messageElement.querySelector('.message-text');
        
        // Clear any existing content
        messageTextEl.innerHTML = '';
        
        // Create a container for the loading dots and text
        const container = document.createElement('div');
        container.className = 'flex items-center';
        
        // Create the dots container
        const dotsContainer = document.createElement('div');
        dotsContainer.className = 'flex space-x-1 mr-2';
        
        // Create three animated dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'w-2 h-2 bg-primary-500 rounded-full animate-bounce';
            dot.style.animationDelay = `${i * 0.2}s`;
            dotsContainer.appendChild(dot);
        }
        
        // Create the text element
        const textSpan = document.createElement('span');
        textSpan.textContent = data.message || 'Analyzing document...';
        
        // Assemble the container
        container.appendChild(dotsContainer);
        container.appendChild(textSpan);
        
        // Add to the message text element
        messageTextEl.appendChild(container);
        
        // Add to messages container
        messagesContainer.appendChild(messageElement);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    /**
     * Handle WebSocket connection established
     * @param {Object} data - Connection data
     */
    function handleConnectionEstablished(data) {
        console.log('Connection confirmed:', data.message);
        updateConnectionStatus('connected');
        
        // Show a welcome toast
        showToast('Connected to AlyaAloft chat server', 'success');
    }
    
    /**
     * Delete a document
     * @param {string} docId - Document ID to delete
     */
    async function deleteDocument(docId) {
        if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/v1/documents/${docId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            // Remove from document list
            appState.documents = appState.documents.filter(doc => doc.id !== docId);
            
            // If this was the selected document, clear the chat
            if (appState.selectedDocument && appState.selectedDocument.id === docId) {
                appState.selectedDocument = null;
                chatTitle.textContent = 'No document selected';
                chatSubtitle.textContent = 'Select a document to start chatting';
                messageInput.disabled = true;
                sendButton.disabled = true;
                messagesContainer.innerHTML = `
                    <div class="flex items-center justify-center h-full text-gray-500">
                        <div class="text-center">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-gray-300 mb-4" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd" />
                            </svg>
                            <p class="text-lg font-medium">No conversation yet</p>
                            <p class="mt-1">Upload or select a document to start chatting</p>
                        </div>
                    </div>
                `;
            }
            
            // Re-render the documents list
            renderDocumentsList();
            
            showToast('Document deleted successfully', 'success');
        } catch (error) {
            console.error('Error deleting document:', error);
            showToast('Failed to delete document', 'error');
        }
    }
    
    /**
     * Show notification to select a document
     */
    function showDocumentSelectionNotification() {
        // Only show if we're on the chat page and the messages container exists
        if (!messagesContainer) return;
        
        // Check if we already have a notification
        if (document.getElementById('document-selection-notification')) return;
        
        // Create notification
        const notification = document.createElement('div');
        notification.id = 'document-selection-notification';
        notification.className = 'fixed bottom-24 right-8 bg-white shadow-lg rounded-lg p-4 max-w-sm w-full z-50 border-l-4 border-primary-500 flex items-start';
        notification.innerHTML = `
            <div class="flex-shrink-0 text-primary-500 mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <div class="flex-1">
                <h3 class="text-gray-800 font-medium mb-1">No document selected</h3>
                <p class="text-gray-600 text-sm">Please upload or select a document from the sidebar to start chatting.</p>
            </div>
            <button id="close-notification" class="flex-shrink-0 text-gray-400 hover:text-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
            </button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Add close event
        document.getElementById('close-notification').addEventListener('click', () => {
            notification.remove();
        });
        
        // Auto-hide after 15 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.add('opacity-0', 'translate-x-10');
                notification.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                
                // Remove after animation
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.remove();
                    }
                }, 500);
            }
        }, 15000);
    }
    
    /**
     * Hide document selection notification
     */
    function hideDocumentSelectionNotification() {
        const notification = document.getElementById('document-selection-notification');
        if (notification) {
            notification.remove();
        }
    }
    
    /**
     * Set up periodic polling for document status updates
     */
    function setupDocumentPolling() {
        // Poll for document status updates every 3 seconds
        setInterval(() => {
            if (appState.documents.length > 0) {
                // Check if any documents are still processing
                const hasProcessingDocuments = appState.documents.some(doc => 
                    doc.processing_status === 'pending' || doc.processing_status === 'processing'
                );
                
                // Only refresh if there are documents still processing
                if (hasProcessingDocuments) {
                    loadDocuments();
                    console.log('Polling for document status updates...');
                }
            }
        }, 3000);
    }
});