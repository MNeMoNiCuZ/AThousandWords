@echo off
echo Starting HTTP server...
echo.
echo Server will be available at: http://localhost:8498
echo Press Ctrl+C to stop the server
echo.

python -m http.server 8498

pause