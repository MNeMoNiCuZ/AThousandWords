# gui/js.py
"""
Javascript injection for the GUI.
"""

JS = r"""
    console.log("[JS] Injecting VRAM click listener (Robust-Selector Version)...");
    
    // Use capture phase on window/documentElement to guarantee we see the event first
    document.documentElement.addEventListener('click', function(e) {
        // Robust targeting using closest() which handles nested clicks automatically
        const target = e.target.closest('.vram-rec-item');
        
        if (target) {
            console.log("[JS] VRAM Item Clicked:", target.innerText);
            
            // Try data attribute first, then fallback to parsing text
            let batchSize = target.getAttribute('data-bs');
            
            if (!batchSize && target.textContent.includes(':')) {
                // Parse "32GB:6" -> "6"
                const match = target.textContent.match(/:(\d+)/);
                if (match) {
                     batchSize = match[1];
                     console.log("[JS] Parsed batch size from text:", batchSize);
                }
            }
            
            if (batchSize) {
                // CRITICAL FIX: Use aria-label selector which is more reliable than ID in Gradio
                const batchInput = document.querySelector('input[aria-label="Batch Size"]');
                
                if (batchInput) {
                    console.log("[JS] Found input, setting value to:", batchSize);
                    
                    batchInput.value = batchSize;
                    
                    // Dispatch all necessary events for Gradio React tracking
                    batchInput.dispatchEvent(new Event('input', { bubbles: true }));
                    batchInput.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // Visual feedback
                    target.style.transition = "transform 0.1s";
                    target.style.transform = "scale(0.95)";
                    setTimeout(() => target.style.transform = "scale(1)", 100);
                    
                } else {
                    console.error("[JS] Batch size input selector 'input[aria-label=\"Batch Size\"]' failed.");
                }
            }
        }
    }, true); // <--- CAPTURE PHASE
    
    // ========================================
    // AUTO-RELOAD ON SESSION MISMATCH (KeyError: 46 fix)
    // ========================================
    // Detect when Gradio's function queue encounters a KeyError (stale function ID)
    // and automatically reload the page ONCE to get fresh session state
    
    (function() {
        let hasReloaded = sessionStorage.getItem('gradio_auto_reload');
        
        // Listen for console errors from Gradio's error messages
        // Intercept fetch errors that indicate queue issues
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            return originalFetch.apply(this, args)
                .then(response => {
                    // Check if response indicates a queue error
                    if (response.url.includes('/queue/join') && !response.ok) {
                        console.warn('[Auto-Reload] Queue join failed, checking for session mismatch...');
                        // Clone response to read body without consuming it
                        response.clone().json().then(data => {
                            if (data && data.error && typeof data.error === 'string') {
                                // Check for KeyError pattern in error message
                                if (data.error.includes('KeyError') || data.error.includes('not found in')) {
                                    console.error('[Auto-Reload] Detected stale Gradio session (function ID mismatch)');
                                    if (!hasReloaded) {
                                        console.log('[Auto-Reload] Reloading page automatically...');
                                        sessionStorage.setItem('gradio_auto_reload', 'true');
                                        setTimeout(() => window.location.reload(), 500);
                                    } else {
                                        console.error('[Auto-Reload] Already reloaded once. Manual refresh required (Ctrl+F5).');
                                        sessionStorage.removeItem('gradio_auto_reload');
                                    }
                                }
                            }
                        }).catch(() => {});
                    }
                    return response;
                })
                .catch(error => {
                    console.error('[Auto-Reload] Fetch error:', error);
                    throw error;
                });
        };
        
        // Clear the reload flag after successful load
        if (hasReloaded) {
            console.log('[Auto-Reload] Page reloaded successfully, clearing flag.');
            setTimeout(() => sessionStorage.removeItem('gradio_auto_reload'), 2000);
        }
    })();
"""
