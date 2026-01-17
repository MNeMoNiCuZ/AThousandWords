# gui/styles.py
"""
CSS styles for the Gradio interface.
"""

CSS = """
/* --- MODERN MODAL POPUP --- */
.modal-overlay { 
    position: fixed !important; 
    top: 0 !important; 
    left: 0 !important; 
    width: 100vw !important; 
    height: 100vh !important; 
    z-index: 2147483647 !important; /* Maximum possible z-index */
    background: rgba(0,0,0,0.85) !important; 
    display: none !important; /* Default hidden */
    justify-content: center !important; 
    align-items: center !important;
    margin: 0 !important; 
    padding: 0 !important;
    pointer-events: auto !important;
}

/* Force visible when Gradio group is visible */
.modal-overlay:not([style*="display: none"]) {
    display: flex !important;
}

.modal-bg-close {
    position: absolute !important; top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1; background: transparent; border: none; cursor: default;
}
.modal-box { 
    background: #111827; padding: 35px; border-radius: 16px; 
    width: 850px; max-height: 85vh; overflow-y: auto; 
    position: relative; z-index: 10001; 
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    border: 1px solid #374151;
    margin: auto !important;
}
.modal-box-wide { 
    background: #111827; padding: 35px; border-radius: 16px; 
    width: 95%; max-width: 1400px; max-height: 85vh; overflow-y: auto; 
    position: relative; z-index: 10001; 
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    border: 1px solid #374151;
    margin: auto !important;
}
.icon-btn { max-width: 40px !important; min-width: 40px !important; width: 40px !important; height: 40px !important; display: flex !important; justify-content: center !important; align-items: center !important; }

/* Ticker Removal - Comprehensive */
/* WebKit browsers */
input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { 
    -webkit-appearance: none !important; margin: 0 !important; display: none !important;
}
/* Firefox */
input[type=number] { -moz-appearance: textfield !important; appearance: textfield !important; }
/* Gradio number input buttons - multiple selectors */
.gr-number-input button, .gradio-number-input button, [class*="number-input"] button,
.gr-box.gr-number button, input[type="number"] + button, input[type="number"] ~ button { 
    display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important;
}

/* Batch Size offset fix - negate larger highlighted VRAM text */
#batch_size {
    margin-top: -8px !important;
}

/* Clickable VRAM recommendations */
.vram-rec-item {
    cursor: pointer !important;
    transition: all 0.2s ease;
    padding: 0px 3px;
    border-radius: 3px;
    display: inline-block; /* Ensure padding works */
    line-height: 1.2; /* Tighter line height */
}
.vram-rec-item:hover {
    background-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-1px);
}

/* Fix Gradio Group/Column remnants */
.modal-overlay > div { background: transparent !important; box-shadow: none !important; border: none !important; }
.modal-overlay.hidden { display: none !important; visibility: hidden !important; height: 0 !important; width: 0 !important; overflow: hidden !important; }

/* Gallery Layout - Left aligned, no cropping */
.gallery-section > div,
.gallery-section > div > div {
    justify-content: flex-start !important;
    align-items: flex-start !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
}
.gallery-section button {
    margin: 2px !important;
}
.gallery-section img {
    object-fit: contain !important;
    background: transparent !important;
}




/* --- MODERN ROUNDED CORNERS --- */
/* Buttons - Modern rounded style with padding */
button.primary,
button.secondary,
button.stop,
.gr-button,
button[class*="btn"],
.gradio-button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
}

/* Groups and Accordions */
.gr-group,
.gr-accordion,
.gradio-group,
.gradio-accordion,
[class*="accordion"],
[class*="group"] {
    border-radius: 12px !important;
}

/* Accordion panels inner content */
.gr-accordion > div,
.gradio-accordion > div {
    border-radius: 0 0 12px 12px !important;
}

/* Text areas and input fields */
textarea,
.gr-text-input,
.gradio-textbox textarea,
[class*="textbox"] textarea {
    border-radius: 8px !important;
}

/* Viewer (Inspector) section styling */
.input-section {
    border-radius: 12px !important;
}

.input-section .gr-group,
.input-section .gradio-group {
    border-radius: 12px !important;
}

/* Tab panels */
.gr-tabs,
.gradio-tabs,
.tabitem,
.tab-nav {
    border-radius: 10px !important;
}

/* Image and video containers in viewer */
.gr-image,
.gr-video,
.gradio-image,
.gradio-video {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Dropdown menus */
.gr-dropdown,
.gradio-dropdown,
select {
    border-radius: 8px !important;
}

/* Number inputs */
.gr-number,
.gradio-number,
input[type="number"] {
    border-radius: 8px !important;
}

/* Checkboxes container */
.gr-checkbox,
.gradio-checkbox {
    border-radius: 4px !important;
}

/* File upload area */
.gr-file,
.gradio-file {
    border-radius: 12px !important;
}

/* Slider track */
.gr-slider,
.gradio-slider {
    border-radius: 8px !important;
}

/* --- Multi-Model Clickable Rows --- */
.clickable-checkbox-row {
    cursor: pointer !important;
    transition: background 0.1s;
    border-radius: 6px;
    padding: 2px 4px;
}
.clickable-checkbox-row:hover {
    background: rgba(255, 255, 255, 0.05);
}
/* Target only the checkbox label within the row */
.clickable-checkbox-row .model-checkbox label {
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}
.clickable-checkbox-row .model-checkbox input[type="checkbox"] {
    cursor: pointer !important;
}

/* --- Download Button --- */
.download-btn {
    background-color: #10b981 !important; /* Emerald 500 */
    color: white !important;
    border: none !important;
    padding: 0 !important;
    min-width: 80px !important;
    width: auto !important; 
    font-size: 1.2em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Ensure the icon image itself is properly sized and centered */
.download-btn img {
    margin: 0 !important;
    display: block !important;
}
.download-btn:hover {
    background-color: #059669 !important; /* Emerald 600 */
}

/* Download button wrapper - minimal size, no expansion */
.download-btn-wrapper {
    display: inline-block !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
}

/* Spinner animation for download button */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.download-btn.processing {
    pointer-events: none;
    box-shadow: none !important; /* Disable pulsing/shadow */
    outline: none !important;
}

/* Hide the icon when processing */
.download-btn.processing img {
    display: none !important;
}

.download-btn.processing::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-top-color: #ffffff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}

/* Prevent any pulsing or shadow on active/focus states - AGGRESSIVE REMOVAL */
.download-btn,
.download-btn button,
.download-btn.primary,
.download-btn:focus,
.download-btn:active,
.download-btn:focus-visible,
.download-btn button:focus,
.download-btn button:focus-visible {
    box-shadow: none !important;
    outline: none !important;
    border-color: transparent !important;
    --ring-color: transparent !important;
    --ring-offset-width: 0px !important;
    --ring-shadow: none !important;
    transition: none !important; /* Stop pulsing animation */
}

/* Pagination Styling */
.pagination-row {
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    height: 32px !important;
    min-height: 0 !important;
    overflow: hidden !important;
    /* Removed display: flex !important to allow hidden state to work */
}

/* When hidden, force height to 0 */
.pagination-row[style*="display: none"] {
    height: 0 !important;
    visibility: hidden !important;
}

/* Force all direct children */
.pagination-row > div,
.pagination-row > button {
    flex: 0 0 auto !important;
    flex-grow: 0 !important;
    width: auto !important;
    min-width: 0 !important;
    margin: 0 !important;
}

/* Compact Buttons - Truly Minimal */
.pagination-btn {
    min-width: 24px !important;
    width: 24px !important;
    height: 24px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    
    /* Strip all button styling */
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    
    color: #9ca3af !important;
    font-weight: bold !important;
    font-size: 1.2em !important; /* Make arrows slightly bigger/clearer */
    line-height: 1 !important;
    cursor: pointer !important;
    
    transform: translateY(-2px) !important; /* Slight lift */
}

/* Remove Gradio's default hover effects if any */
.pagination-btn:hover {
    background: transparent !important;
    background-color: rgba(255, 255, 255, 0.1) !important; /* Only subtle bg on hover */
    color: #ffffff !important;
    box-shadow: none !important;
    border-radius: 4px !important;
}

/* Compact Input - Force Width */
.pagination-input {
    width: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
    height: 24px !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* Input field inside wrapper - Highly Specific */
.pagination-input input, 
.pagination-input input[type="number"] {
    text-align: center !important;
    padding: 0 !important;
    height: 24px !important;
    font-size: 0.95em !important;
    font-weight: bold !important;
    color: #e5e7eb !important; /* Bright text */
    background: transparent !important; /* Blend in completely */
    background-color: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    outline: none !important;
    -moz-appearance: textfield !important;
}
/* Hide spin buttons */
.pagination-input input::-webkit-outer-spin-button,
.pagination-input input::-webkit-inner-spin-button {
  -webkit-appearance: none !important;
  margin: 0 !important;
}

.pagination-input input:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
}
.pagination-input input:focus {
    background: rgba(31, 41, 55, 1.0) !important;
    background-color: rgba(31, 41, 55, 1.0) !important;
    border-color: #4b5563 !important;
}

/* Compact Label */
.pagination-label {
    min-width: unset !important;
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 24px !important;
}
.pagination-label p {
    margin: 0 !important;
    padding: 0 4px !important;
    font-size: 0.9em !important;
    font-weight: bold !important;
    color: #6b7280 !important;
    white-space: nowrap !important;
}

/* Model Description Display */
.model-description {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.model-description p {
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    font-size: 0.95em !important;
    font-style: italic !important;
    color: #9ca3af !important;
    line-height: 1.3 !important;
}

/* --- Presets Table Styling (Matches Model Info) --- */
.preset-header {
    border-bottom: 2px solid #555;
    margin-bottom: 0 !important;
    font-weight: bold;
    font-size: 1.1em;
    padding-bottom: 10px;
}

.preset-row {
    border-bottom: 1px solid #374151; /* #444 equivalent-ish */
    align-items: center !important;
    padding: 0 !important;
    min-height: 48px; /* Slightly taller for padding */
    align-content: center;
    transition: background 0.1s;
}

.preset-row:hover {
    background: rgba(255,255,255,0.03);
}

.preset-cell {
    padding: 10px; /* Exact match to model_info padding */
    font-size: 1.0em;
    display: flex;
    align-items: center;
    height: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Delete Icon Button - Minimal */
.table-icon-btn {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1.4em !important; /* Larger icon */
    padding: 0 !important;
    margin: 0 !important;
    width: 40px !important; /* Fixed small width */
    height: 40px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    cursor: pointer !important;
    opacity: 0.7;
    color: #ef4444 !important;
}
.table-icon-btn:hover {
    background: rgba(239, 68, 68, 0.1) !important; 
    opacity: 1.0;
    border-radius: 4px;
}
"""
