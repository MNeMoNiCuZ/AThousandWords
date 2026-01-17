"""
Image Cropping Tool

Embeds the interactive Angular-based image cropping web app via iframe.
The web app provides visual crop previews with rectangles and focal point adjustment.
"""

import logging
import threading
import socket
from pathlib import Path
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import gradio as gr

from .base import BaseTool, ToolConfig


class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses request logging."""
    
    def log_message(self, format, *args):
        pass


class CroppingTool(BaseTool):
    """Tool that embeds the interactive image cropping web app."""
    
    _server = None
    _server_thread = None
    _port = None
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="cropping",
            display_name="Crop Images",
            description="""### Interactive Image Cropping
Visual batch cropping tool with smart focal point detection.
Click 'Launch Cropping App' to start the embedded cropping interface.""",
            icon=""
        )
    
    def apply_to_dataset(self, dataset, **kwargs) -> str:
        """Start server and return status."""
        port = self._ensure_server_running()
        if port:
            return f"Cropping app running at http://localhost:{port}"
        return "Failed to start cropping server"
    
    def create_gui(self, app) -> tuple:
        """Create the Crop tool UI with launch button."""
        
        gr.Markdown(self.config.description)
        
        self.iframe_output = gr.HTML(
            value='<div style="padding:20px; color:#888; background:#1a1a2e; border-radius:8px;">Click "Launch Cropping App" to start the interactive cropping interface.</div>',
            elem_id="crop_iframe_container"
        )
        
        launch_btn = gr.Button("Launch Cropping App", variant="primary", elem_id="crop_launch_btn")
        
        return (launch_btn, [])
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        """Wire launch button to start server and show iframe."""
        
        def launch_app():
            port = self._ensure_server_running()
            if port:
                iframe_html = f'''
<div style="width:100%; height:800px; border:1px solid #444; border-radius:8px; overflow:hidden;">
    <iframe 
        src="http://localhost:{port}/" 
        style="width:100%; height:100%; border:none;"
        allow="clipboard-read; clipboard-write"
    ></iframe>
</div>
<p style="color:#888; font-size:12px; margin-top:8px;">
    Cropping app running at <a href="http://localhost:{port}/" target="_blank" style="color:#64B5F6;">localhost:{port}</a> - 
    <a href="http://localhost:{port}/" target="_blank" style="color:#64B5F6;">Open in new tab</a>
</p>
'''
                gr.Info(f"Cropping app started at localhost:{port}")
                return iframe_html
            else:
                gr.Warning("Failed to start cropping server")
                return '<div style="padding:20px; color:#f87171;">Failed to start server.</div>'
        
        run_button.click(fn=launch_app, inputs=[], outputs=[self.iframe_output])
    
    def _ensure_server_running(self) -> int:
        """Start HTTP server if not running. Returns port number."""
        if CroppingTool._server is not None and CroppingTool._port is not None:
            return CroppingTool._port
        
        try:
            static_dir = Path(__file__).parent / "image_cropping"
            if not static_dir.exists():
                logging.error(f"Cropping app directory not found: {static_dir}")
                return None
            
            port = self._find_available_port()
            if port is None:
                logging.error("Could not find available port for cropping server")
                return None
            
            handler = partial(QuietHTTPHandler, directory=str(static_dir))
            server = HTTPServer(("127.0.0.1", port), handler)
            
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            
            CroppingTool._server = server
            CroppingTool._server_thread = thread
            CroppingTool._port = port
            
            logging.info(f"Cropping server started at http://localhost:{port}")
            print(f"\033[96mCropping server started at http://localhost:{port}\033[0m")
            
            return port
            
        except Exception as e:
            logging.error(f"Failed to start cropping server: {e}")
            return None
    
    def _find_available_port(self, start=8498, end=8599) -> int:
        """Find an available port in range."""
        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        return None
