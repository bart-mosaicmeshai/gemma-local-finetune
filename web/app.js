// Bluey AI Chatbot - Browser-based inference with Transformers.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3';

// Configuration
const CONFIG = {
    modelPath: './',  // Base path where models folder is located
    modelId: 'models',  // Model folder name
    maxNewTokens: 100,
    temperature: 0.8,
    topP: 0.95,
    repetitionPenalty: 1.1,
};

// Global state
let generator = null;
let isGenerating = false;

// UI Elements
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const buttonText = document.getElementById('buttonText');
const buttonSpinner = document.getElementById('buttonSpinner');

// Update status
function updateStatus(message, type = 'info') {
    statusText.textContent = message;
    statusIndicator.className = `status-indicator ${type}`;
}

// Add message to chat
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '👤' : '🐕';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);

    // Remove welcome message if it exists
    const welcomeMessage = chatContainer.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Initialize the model
async function initializeModel() {
    try {
        updateStatus('Loading model... This may take a few minutes on first load.', 'loading');

        // Configure transformers.js for local models
        env.allowLocalModels = true;
        env.localModelPath = CONFIG.modelPath;

        // Check for WebGPU support
        const hasWebGPU = !!navigator.gpu;
        const device = hasWebGPU ? 'webgpu' : 'wasm';

        updateStatus(`Loading model using ${device}...`, 'loading');

        // Load the text generation pipeline (matching Google example structure)
        generator = await pipeline(
            'text-generation',
            CONFIG.modelId,
            {
                dtype: 'q4',
                device: device,
                model_file_name: 'model',
            }
        );

        updateStatus('Model loaded! Ready to chat.', 'ready');
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();

    } catch (error) {
        console.error('Error loading model:', error);
        console.error('Error stack:', error.stack);
        console.error('Error details:', JSON.stringify(error, null, 2));
        updateStatus(`Error: ${error.message || error}. See console for details.`, 'error');

        // Show helpful error message
        addMessage(
            `Sorry, I couldn't load the model. Error: ${error.message || error}. ` +
            `See the README.md in the web folder for setup instructions.`,
            false
        );
    }
}

// Generate response
async function generateResponse(prompt) {
    if (!generator || isGenerating) return;

    isGenerating = true;
    sendButton.disabled = true;
    buttonText.style.display = 'none';
    buttonSpinner.style.display = 'inline-block';

    try {
        // Format prompt for Bluey personality
        const formattedPrompt = `User: ${prompt}\nBluey:`;

        // Generate response
        const output = await generator(formattedPrompt, {
            max_new_tokens: CONFIG.maxNewTokens,
            temperature: CONFIG.temperature,
            top_p: CONFIG.topP,
            repetition_penalty: CONFIG.repetitionPenalty,
            do_sample: true,
        });

        // Extract the generated text
        let response = output[0].generated_text;

        // Remove the prompt from the response
        response = response.replace(formattedPrompt, '').trim();

        // Clean up response
        // 1. Take only the first line (before any newline)
        response = response.split('\n')[0].trim();

        // 2. Remove trailing JSON-like artifacts (brackets, quotes, etc.)
        response = response.replace(/["\]}\[{]+$/g, '').trim();

        // 3. Remove leading special characters
        response = response.replace(/^["\]}\[{]+/g, '').trim();

        // Add bot response
        addMessage(response || "G'day! Can you ask that again?", false);

    } catch (error) {
        console.error('Error generating response:', error);
        addMessage('Oops! Something went wrong. Try asking again!', false);
    } finally {
        isGenerating = false;
        sendButton.disabled = false;
        buttonText.style.display = 'inline';
        buttonSpinner.style.display = 'none';
        userInput.focus();
    }
}

// Handle send message
async function handleSend() {
    const message = userInput.value.trim();
    if (!message || isGenerating) return;

    // Add user message
    addMessage(message, true);
    userInput.value = '';

    // Generate response
    await generateResponse(message);
}

// Event listeners
sendButton.addEventListener('click', handleSend);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
});

// Initialize on load
window.addEventListener('load', () => {
    initializeModel();
});
