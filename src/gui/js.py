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
"""
