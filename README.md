# Scenario Image Automation GUI

This application allows you to create, edit, and run automation scenarios based on image detection on your screen. Built with Python and PyQt6, it supports:

- Scenario creation, selection, and management
- Drawing screenshots for image recognition
- Defining actions (click, key, scroll) with position and delay options
- Start/Stop controls with hotkey support
- Import/export of scenario configuration files
- Auto-save on scenario changes

## Requirements
- Python 3.8+
- PyQt6
- opencv-python
- pyautogui
- pynput

## Running the App
1. Install dependencies (already handled by setup)
2. Run:
   ```
   C:/Users/GabrielFranz/Documents/VS Code/Test/.venv/Scripts/python.exe main.py
   ```

## Notes
- All scenario changes are saved automatically.
- Hotkeys can be configured in the settings.
- For best results, use high-quality reference images for detection.
