# VisionFlow Automator

VisionFlow Automator is a powerful desktop automation tool that lets you create, edit, and run automation scenarios based on image recognition and GUI actions. With an intuitive graphical interface, you can visually define workflows that interact with any application or window on your system.

## Features

- **Scenario Management**: Create, edit, rename, import, export, and organize automation scenarios.
- **Step-Based Automation**: Each scenario consists of multiple steps, each with its own set of images and actions.
- **Screenshot-Based Image Matching**: Take region-based screenshots as image triggers for steps. Adjustable sensitivity for robust detection.
- **Flexible Actions**: Automate clicks (with position modes), key presses, scrolling, and delays. Actions can be chained per step.
- **Window Targeting**: Run automations on the entire screen or a specific open window.
- **Hotkey Control**: Start/stop automation instantly with a global hotkey (default: F9).
- **Visual Step Editor**: Add, edit, reorder, and preview step images and actions with a modern PyQt6 interface.
- **Import/Export**: Share or back up scenarios as ZIP files.
- **Persistent State**: Auto-saves all changes and remembers your last scenario and window selection.
- **Logging**: Detailed logs for troubleshooting and auditing automation runs.

## How It Works

1. **Create a Scenario**: Define a new scenario and add steps.
2. **Add Step Images**: For each step, take a screenshot of the region to match. Multiple images per step are supported.
3. **Define Actions**: Assign actions (click, key, scroll, delay) to perform when images are detected.
4. **Set Conditions**: Choose whether all images (AND) or any image (OR) must be detected to trigger actions.
5. **Target Window**: Select the window or use the entire screen for automation.
6. **Run Automation**: Start/stop with the hotkey or UI button. The app continuously looks for image matches and performs actions as defined.

## Requirements

- Python 3.11+
- PyQt6
- opencv-python
- pyautogui
- pynput
- pygetwindow
- numpy
- Pillow

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the App

1. Ensure all dependencies are installed.
2. Run the main script:
   ```bash
   python main.py
   ```

## Tips
- Use high-quality, tightly-cropped reference images for best detection results.
- Adjust sensitivity per image for robust matching.
- Use the import/export feature to share scenarios between machines.
- All changes are saved automatically; no need to manually save.

## License
MIT License

---

VisionFlow Automator — Automate anything you can see!