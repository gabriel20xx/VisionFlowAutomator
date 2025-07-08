import sys
import os
import json
import threading
import time
import logging
import zipfile
import shutil
from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import pygetwindow as gw
import numpy as np
import pyautogui
from pynput import keyboard
import tkinter as tk
from PIL import ImageGrab, Image, ImageTk


# Setup logs directory
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'scenario_automation.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_DIR = 'scenarios'
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

class Scenario:
    """
    Represents a scenario, which consists of a name and a list of steps.
    Handles saving/loading scenario data to/from disk.
    """
    def __init__(self, name, steps=None):
        """
        Initialize a Scenario with a name and optional steps.
        """
        self.name = name
        self.steps = steps or []  # List of Step dicts

    def to_dict(self):
        """
        Return a dictionary representation of the scenario.
        """
        return {'name': self.name, 'steps': self.steps}

    @staticmethod
    def from_dict(data):
        """
        Create a Scenario object from a dictionary.
        """
        return Scenario(data['name'], data.get('steps', []))

    def get_scenario_dir(self):
        """
        Return the directory path for this scenario's files.
        """
        return os.path.join(CONFIG_DIR, self.name)

    def save(self):
        """
        Save the scenario to disk as a JSON file.
        """
        try:
            scenario_dir = self.get_scenario_dir()
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)
            
            with open(os.path.join(scenario_dir, 'scenario.json'), 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Scenario '{self.name}' saved.")
        except Exception as e:
            logger.error(f"Failed to save scenario '{self.name}': {e}")

    @staticmethod
    def load(name):
        """
        Load a scenario from disk by name.
        """
        try:
            scenario_dir = os.path.join(CONFIG_DIR, name)
            with open(os.path.join(scenario_dir, 'scenario.json'), 'r') as f:
                scenario = Scenario.from_dict(json.load(f))
            logger.info(f"Scenario '{name}' loaded.")
            return scenario
        except Exception as e:
            logger.error(f"Failed to load scenario '{name}': {e}")
            return None

    @staticmethod
    def list_all():
        """
        List all scenario directories.
        """
        return [d for d in os.listdir(CONFIG_DIR) if os.path.isdir(os.path.join(CONFIG_DIR, d))]

def take_screenshot_with_tkinter():
    """
    Use Tkinter to let the user select a region of the screen for a screenshot.
    Returns a dict with x, y, width, height.
    """
    root = tk.Tk()
    root.attributes("-alpha", 0.3)
    root.attributes("-fullscreen", True)
    root.wait_visibility(root)
    
    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill="both", expand=True)

    rect = None
    start_x = None
    start_y = None
    
    im = ImageGrab.grab()
    img = ImageTk.PhotoImage(im)
    canvas.create_image(0,0,image=img,anchor="nw")

    selection_rect = {"x": 0, "y": 0, "width": 0, "height": 0}

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x = event.x
        start_y = event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

    def on_mouse_drag(event):
        nonlocal rect
        cur_x, cur_y = (event.x, event.y)
        canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def on_button_release(event):
        nonlocal selection_rect
        end_x, end_y = (event.x, event.y)
        
        selection_rect["x"] = min(start_x, end_x)
        selection_rect["y"] = min(start_y, end_y)
        selection_rect["width"] = abs(start_x - end_x)
        selection_rect["height"] = abs(start_y - end_y)
        
        root.quit()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    root.mainloop()
    root.destroy()
    
    return selection_rect

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for Scenario Image Automation.
    Handles UI, scenario management, and automation logic.
    """
    def automation_loop(self):
        """
        Main loop for running automation steps. Continuously looks for image matches
        in the selected window or screen and performs actions as defined in the scenario.
        """
        logger.info('Automation loop started.')
        try:
            while self.running:
                # Only update the state label, do not log every loop
                self.set_state('Looking for matches')
                selected_window = self.window_combo.currentText()
                if selected_window == 'Entire Screen':
                    screen = pyautogui.screenshot()
                    offset_x, offset_y = 0, 0
                else:
                    try:
                        win = None
                        for w in gw.getAllWindows():
                            if w.title == selected_window:
                                win = w
                                break
                        if win is not None and win.width > 0 and win.height > 0:
                            x, y, w_, h_ = win.left, win.top, win.width, win.height
                            screen = pyautogui.screenshot(region=(x, y, w_, h_))
                            offset_x, offset_y = x, y
                        else:
                            screen = pyautogui.screenshot()
                            offset_x, offset_y = 0, 0
                    except Exception as e:
                        logger.error(f'Error capturing window screenshot: {e}')
                        screen = pyautogui.screenshot()
                        offset_x, offset_y = 0, 0
                screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
                for step in self.current_scenario.steps:
                    # Detect all images for this step
                    detections = {}
                    for img in step.get('images', []):
                        try:
                            template = cv2.imread(img['path'])
                            if template is None:
                                logger.warning(f"Could not load template image: {img['path']}")
                                continue
                            res = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            # logger.debug(f"Detection for {img['name']}: max_val={max_val}")  # Suppressed per request
                            sensitivity = img.get('sensitivity', 0.9)
                            if max_val > sensitivity:
                                # Adjust detected location to be relative to the full screen
                                abs_loc = (max_loc[0] + offset_x, max_loc[1] + offset_y)
                                detections[img['name']] = (abs_loc, template.shape)
                        except Exception as e:
                            logger.error(f"Error in image detection for {img['name']}: {e}")
                    # Check condition
                    cond = step.get('condition', 'OR')
                    found = [img for img in step.get('images', []) if img.get('name') in detections]
                    trigger = False
                    if not step.get('images', []):
                        trigger = False
                    elif cond == 'AND':
                        trigger = len(found) == len(step.get('images', []))
                    else:
                        trigger = len(found) > 0
                    # logger.debug(f"Step '{step.get('name', 'step')}' trigger check: found={found}, cond={cond}, trigger={trigger}")  # Suppressed per request
                    if trigger:
                        self.set_state(f'Performing step: {step.get("name", "step")})')
                        logger.info(f"Found match for step: {step.get('name', 'step')}. Performing actions.")
                        # Use the first detected image for position
                        ref_img = found[0] if found else step.get('images', [])[0]
                        loc, shape = detections.get(ref_img.get('name'), ((0, 0), (0, 0, 0)))
                        for act in step.get('actions', []):
                            self._perform_step_action(act, loc, shape)
                        logger.info(f"Performed actions for step: {step.get('name', 'step')}")
                        time.sleep(1)  # Prevent spamming
                time.sleep(0.2)
        except Exception as e:
            logger.error(f'Automation loop error: {e}')
        finally:
            self.set_state('Paused')

    def _perform_step_action(self, action, loc, shape):
        """
        Perform a single step action (click, key, scroll, delay) at the given location.
        """
        act_type = action['type']
        params = action['params']
        logger.debug(f'Performing step action: {act_type}, params={params}, loc={loc}, shape={shape}')
        try:
            if act_type == 'click':
                pos_type = params.get('pos_type', 'center')
                if pos_type == 'center':
                    x = loc[0] + shape[1] // 2
                    y = loc[1] + shape[0] // 2
                elif pos_type == 'relative':
                    x = loc[0] + params.get('rel_x', 0)
                    y = loc[1] + params.get('rel_y', 0)
                elif pos_type == 'absolute':
                    x = params.get('abs_x', 0)
                    y = params.get('abs_y', 0)
                else:
                    x, y = loc[0], loc[1]
                button = params.get('button', 'left')
                logger.info(f'Clicking at ({x}, {y}) with button {button}')
                pyautogui.click(x, y, button=button)
            elif act_type == 'key':
                key = params.get('key', 'enter')
                logger.info(f'Pressing key: {key}')
                pyautogui.press(key)
            elif act_type == 'scroll':
                amount = params.get('amount', 0)
                direction = params.get('direction', 'up')
                logger.info(f'Scrolling: {amount} direction: {direction}')
                if direction == 'up':
                    pyautogui.scroll(amount)
                else:
                    pyautogui.scroll(-abs(amount))
            elif act_type == 'delay':
                duration = params.get('duration', 1)
                logger.info(f'Delay for {duration} seconds')
                time.sleep(duration)
        except Exception as e:
            logger.error(f'Error performing step action {act_type}: {e}')
    def set_state(self, text):
        """
        Update the state label in the UI.
        """
        self.state_text = text
        self.state_label.setText(f'State: {text}')

    def toggle_automation(self):
        """
        Toggle the automation state (start/stop). A debounce delay is added only
        when changing from stop to start, not from start to stop.
        """
        if self.running:
            self.stop_automation()
        else:
            now = time.time()
            if now - self.last_toggle_time < 5:
                logger.info("Start toggle ignored: pressed too quickly after stopping.")
                QtWidgets.QMessageBox.information(self, "Wait", "Please wait at least 5 seconds before starting again.")
                return
            self.start_automation()

    def start_automation(self):
        """
        Start the automation process and worker thread.
        """
        logger.info('Starting automation.')
        if not self.current_scenario or self.running:
            logger.warning('Start Automation: No scenario selected or already running.')
            return
        self.running = True
        self.set_state('Looking for matches')
        self.btn_start_stop.setText(f'Stop ({self.hotkey.upper()})')
        self.worker = threading.Thread(target=self.automation_loop, daemon=True)
        self.worker.start()
        self.listener = keyboard.GlobalHotKeys({self.hotkey: self.stop_automation})
        self.listener.start()

    def stop_automation(self):
        """
        Stop the automation process and worker thread.
        """
        logger.info('Stopping automation.')
        self.running = False
        self.last_toggle_time = time.time()
        self.set_state('Paused')
        self.btn_start_stop.setText(f'Start ({self.hotkey.upper()})')
        if self.listener:
            self.listener.stop()
            self.listener = None

    def __init__(self):
        """
        Initialize the main window, UI, and load scenarios.
        """
        super().__init__()
        self.setWindowTitle('Scenario Image Automation')
        # Make window smaller and more compact
        min_width = 480
        min_height = 480
        self.setMinimumSize(min_width, min_height)
        self.setGeometry(100, 100, min_width, min_height)
        self.running = False
        self.hotkey = '<f9>'
        self.listener = None
        self.worker = None
        self.current_scenario = None
        self.selected_step_idx = None
        self.last_toggle_time = 0  # For debounce of start/stop
        self.init_ui()
        self.load_scenarios()

    def init_ui(self):
        """
        Initializes the main UI components and layouts.
        Sets up scenario and window dropdowns, step list, and action buttons.
        """
        # Set compact font and style
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)
        style = """
            QPushButton, QComboBox, QListWidget, QLabel {
                font-size: 9pt;
                min-height: 20px;
                padding: 2px 6px;
            }
            QComboBox { min-width: 90px; }
            QPushButton { min-width: 70px; }
            QListWidget { min-width: 200px; min-height: 120px; }
        """
        self.setStyleSheet(style)

        # Scenario group (QGroupBox with title 'Scenario')
        scenario_group_box = QtWidgets.QGroupBox('Scenario')
        scenario_group_box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        scenario_group_layout = QtWidgets.QVBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(90)
        self.combo.currentIndexChanged.connect(self._log_combo_scenario)
        scenario_group_layout.addWidget(self.combo)

        # Window selection dropdown
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setMinimumWidth(90)
        # --- Window selection group ---
        window_group_box = QtWidgets.QGroupBox('Target Window')
        window_group_box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        window_group_layout = QtWidgets.QHBoxLayout()
        self.window_combo.setMinimumWidth(90)
        self.window_combo.currentIndexChanged.connect(self._log_combo_window)
        self.btn_refresh_windows = QtWidgets.QPushButton('Refresh')
        self.btn_refresh_windows.setToolTip('Refresh open windows list')
        self.btn_refresh_windows.clicked.connect(self._log_btn_refresh_windows)
        window_group_layout.addWidget(self.window_combo)
        window_group_layout.addWidget(self.btn_refresh_windows)
        window_group_box.setLayout(window_group_layout)
        self.refresh_window_list()
        # Scenario actions
        self.btn_new = QtWidgets.QPushButton('New')
        self.btn_new.setToolTip('Create new scenario')
        self.btn_new.clicked.connect(self._log_btn_new)
        self.btn_import = QtWidgets.QPushButton('Import')
        self.btn_import.setToolTip('Import scenario')
        self.btn_import.clicked.connect(self._log_btn_import)
        self.btn_export = QtWidgets.QPushButton('Export')
        self.btn_export.setToolTip('Export scenario')
        self.btn_export.clicked.connect(self._log_btn_export)
        self.btn_rename_scenario = QtWidgets.QPushButton("Rename")
        self.btn_rename_scenario.setToolTip('Rename scenario')
        self.btn_rename_scenario.clicked.connect(self._log_btn_rename_scenario)
        scenario_btn_group_layout = QtWidgets.QHBoxLayout()
        scenario_btn_group_layout.setSpacing(4)
        scenario_btn_group_layout.setContentsMargins(4, 4, 4, 4)
        scenario_btn_group_layout.addWidget(self.btn_new)
        scenario_btn_group_layout.addWidget(self.btn_import)
        scenario_btn_group_layout.addWidget(self.btn_export)
        scenario_btn_group_layout.addWidget(self.btn_rename_scenario)
        scenario_group_layout.addLayout(scenario_btn_group_layout)
        scenario_group_box.setLayout(scenario_group_layout)

        # Steps group (QGroupBox with title 'Steps')
        steps_group_box = QtWidgets.QGroupBox('Steps')
        steps_group_layout = QtWidgets.QVBoxLayout()
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.setMinimumHeight(120)
        self.steps_list.currentRowChanged.connect(self._log_steps_list)
        steps_group_layout.addWidget(self.steps_list)
        # Step actions
        self.btn_add_step = QtWidgets.QPushButton('Add')
        self.btn_add_step.setToolTip('Add step')
        self.btn_add_step.clicked.connect(self._log_btn_add_step)
        self.btn_edit_step = QtWidgets.QPushButton('Edit')
        self.btn_edit_step.setToolTip('Edit step')
        self.btn_edit_step.clicked.connect(self._log_btn_edit_step)
        self.btn_del_step = QtWidgets.QPushButton('Delete')
        self.btn_del_step.setToolTip('Delete step')
        self.btn_del_step.clicked.connect(self._log_btn_del_step)
        self.btn_rename_step = QtWidgets.QPushButton("Rename")
        self.btn_rename_step.setToolTip('Rename step')
        self.btn_rename_step.clicked.connect(self._log_btn_rename_step)
        step_btn_group_layout = QtWidgets.QHBoxLayout()
        step_btn_group_layout.setSpacing(4)
        step_btn_group_layout.setContentsMargins(4, 4, 4, 4)
        step_btn_group_layout.addWidget(self.btn_add_step)
        step_btn_group_layout.addWidget(self.btn_edit_step)
        step_btn_group_layout.addWidget(self.btn_del_step)
        step_btn_group_layout.addWidget(self.btn_rename_step)
        steps_group_layout.addLayout(step_btn_group_layout)
        steps_group_box.setLayout(steps_group_layout)

        # Start/Stop Combined
        self.btn_start_stop = QtWidgets.QPushButton(f'Start ({self.hotkey.upper()})')
        self.btn_start_stop.setMinimumWidth(80)
        self.btn_start_stop.clicked.connect(self._log_btn_start_stop)

        # State label
        self.state_label = QtWidgets.QLabel('State: Paused')
        self.state_label.setStyleSheet('font-weight: bold; font-size: 11pt; color: #0055aa;')
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Main layout
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        # Scenario, Window, and Steps groups in layout
        layout.addWidget(scenario_group_box, 0, 0, 1, 5)
        layout.setRowStretch(0, 0)  # Scenario group should not stretch vertically
        layout.addWidget(window_group_box, 1, 0, 1, 5)
        layout.addWidget(steps_group_box, 2, 0, 3, 5)

        # Start/State row in a group
        start_state_group = QtWidgets.QGroupBox()
        start_state_group.setTitle("")
        start_state_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        start_state_layout = QtWidgets.QHBoxLayout()
        start_state_layout.setSpacing(6)  # Fixed gap between button and label
        start_state_layout.addWidget(self.btn_start_stop)
        start_state_layout.addWidget(self.state_label)
        start_state_layout.addStretch(1)
        start_state_group.setLayout(start_state_layout)
        layout.addWidget(start_state_group, 5, 0, 1, 5)
        layout.setRowStretch(5, 0)  # Prevent vertical stretch

        # Set central widget (must be at the end of init_ui)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _log_btn_refresh_windows(self):
        logger.info("UI: Refresh Windows button pressed")
        self.refresh_window_list()

    def _log_combo_scenario(self, idx):
        """
        Log scenario combo box change and call select_scenario.
        """
        logger.info(f"UI: Scenario combo changed to index {idx} ({self.combo.currentText()})")
        self.select_scenario()

    def _log_combo_window(self, idx):
        """
        Log window combo box change and call save_selected_window.
        """
        logger.info(f"UI: Window combo changed to index {idx} ({self.window_combo.currentText()})")
        self.save_selected_window()

    def _log_btn_new(self):
        """
        Log New Scenario button press and call create_scenario.
        """
        logger.info("UI: New Scenario button pressed")
        self.create_scenario()

    def _log_btn_import(self):
        """
        Log Import Scenario button press and call import_scenario.
        """
        logger.info("UI: Import Scenario button pressed")
        self.import_scenario()

    def _log_btn_export(self):
        """
        Log Export Scenario button press and call export_scenario.
        """
        logger.info("UI: Export Scenario button pressed")
        self.export_scenario()

    def _log_btn_rename_scenario(self):
        """
        Log Rename Scenario button press and call rename_scenario.
        """
        logger.info("UI: Rename Scenario button pressed")
        self.rename_scenario()

    def _log_steps_list(self, idx):
        """
        Log steps list row change and call select_step.
        """
        logger.info(f"UI: Steps list row changed to {idx}")
        self.select_step(idx)

    def _log_btn_add_step(self):
        """
        Log Add Step button press and call add_step.
        """
        logger.info("UI: Add Step button pressed")
        self.add_step()

    def _log_btn_edit_step(self):
        """
        Log Edit Step button press and call edit_step.
        """
        logger.info("UI: Edit Step button pressed")
        self.edit_step()

    def _log_btn_del_step(self):
        """
        Log Delete Step button press and call delete_step.
        """
        logger.info("UI: Delete Step button pressed")
        self.delete_step()

    def _log_btn_rename_step(self):
        """
        Log Rename Step button press and call rename_step.
        """
        logger.info("UI: Rename Step button pressed")
        self.rename_step()

    def _log_btn_start_stop(self):
        """
        Log Start/Stop button press and call toggle_automation.
        """
        logger.info("UI: Start/Stop button pressed")
        self.toggle_automation()

    def select_step(self, idx):
        """
        Set the selected step index in the UI.
        """
        self.selected_step_idx = idx

    def add_step(self):
        """
        Open the dialog to add a new step to the current scenario.
        """
        self.setWindowState(QtCore.Qt.WindowState.WindowMinimized)
        try:
            dlg = StepDialog(self)
            dlg.setWindowFlags(dlg.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            # Only process if the dialog was accepted
            if dlg.exec():
                step = dlg.get_step()
                self.current_scenario.steps.append(step)
                self.current_scenario.save()
                self.refresh_lists()
                logger.info(f'Added step: {step}')
        finally:
            self.setWindowState(QtCore.Qt.WindowState.WindowNoState)
            self.activateWindow()

    def edit_step(self):
        """
        Open the dialog to edit the selected step in the current scenario.
        """
        idx = self.selected_step_idx
        if idx is None or idx < 0 or idx >= len(self.current_scenario.steps):
            return
        step = self.current_scenario.steps[idx]
        self.setWindowState(QtCore.Qt.WindowState.WindowMinimized)
        try:
            dlg = StepDialog(self, step)
            dlg.setWindowFlags(dlg.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
            # Only process if the dialog was accepted
            if dlg.exec():
                self.current_scenario.steps[idx] = dlg.get_step()
                self.current_scenario.save()
                self.refresh_lists()
                logger.info(f'Edited step at idx {idx}')
        finally:
            self.setWindowState(QtCore.Qt.WindowState.WindowNoState)
            self.activateWindow()

    def delete_step(self):
        """
        Delete the selected step from the current scenario.
        """
        idx = self.selected_step_idx
        if idx is None or idx < 0 or idx >= len(self.current_scenario.steps):
            return
        del self.current_scenario.steps[idx]
        self.current_scenario.save()
        self.refresh_lists()
        logger.info(f'Deleted step at idx {idx}')

    def save_last_scenario(self, name):
        """
        Save the name of the last selected scenario to disk.
        """
        try:
            with open(os.path.join(CONFIG_DIR, 'last_scenario.txt'), 'w') as f:
                f.write(name)
        except Exception as e:
            logger.error(f"Could not save last scenario: {e}")

    def read_last_scenario(self):
        """
        Read the name of the last selected scenario from disk.
        """
        try:
            path = os.path.join(CONFIG_DIR, 'last_scenario.txt')
            if not os.path.exists(path):
                return None
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Could not read last scenario: {e}")
            return None

    def load_scenarios(self):
        """
        Load all scenarios and populate the scenario combo box.
        """
        logger.debug('Loading scenarios...')
        self.combo.clear()
        scenarios = Scenario.list_all()
        for name in scenarios:
            self.combo.addItem(name)

        self.refresh_window_list()

        if not scenarios:
            choice_dialog = QtWidgets.QMessageBox(self)
            choice_dialog.setWindowTitle('No Scenarios Found')
            choice_dialog.setText('No scenarios found. Would you like to create a new scenario or import one?')
            create_btn = choice_dialog.addButton('Create New', QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            import_btn = choice_dialog.addButton('Import', QtWidgets.QMessageBox.ButtonRole.ActionRole)
            choice_dialog.setDefaultButton(create_btn)
            choice_dialog.exec()
            clicked = choice_dialog.clickedButton()
            if clicked == create_btn:
                name, ok = QtWidgets.QInputDialog.getText(self, 'New Scenario', 'Enter a name for your first scenario:')
                if ok and name:
                    logger.info(f'Creating new scenario: {name}')
                    s = Scenario(name)
                    s.save()
                    self.combo.addItem(name)
                    self.combo.setCurrentText(name)
                else:
                    self.refresh_lists()
            elif clicked == import_btn:
                self.import_scenario()
            else:
                self.refresh_lists()
        else:
            last_scenario_name = self.read_last_scenario()
            if last_scenario_name and last_scenario_name in scenarios:
                self.combo.setCurrentText(last_scenario_name)
            else:
                self.combo.setCurrentIndex(0)

    def select_scenario(self):
        """
        Load the selected scenario and update the UI.
        """
        name = self.combo.currentText()
        logger.info(f'Scenario selected: {name}')
        if name:
            self.current_scenario = Scenario.load(name)
            self.save_last_scenario(name)
            self.refresh_lists()
            self.load_selected_window_from_config()

    def create_scenario(self):
        """
        Open a dialog to create a new scenario.
        """
        name, ok = QtWidgets.QInputDialog.getText(self, 'New Scenario', 'Enter scenario name:')
        if ok and name:
            logger.info(f'Creating new scenario: {name}')
            s = Scenario(name)
            s.save()
            self.load_scenarios()
            self.combo.setCurrentText(name)

    def rename_scenario(self):
        """
        Open a dialog to rename the current scenario.
        """
        if not self.current_scenario:
            return
        
        old_name = self.current_scenario.name
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Scenario", "New name:", text=old_name)
        
        if ok and new_name and new_name != old_name:
            # Rename the scenario directory
            try:
                old_dir = os.path.join(CONFIG_DIR, old_name)
                new_dir = os.path.join(CONFIG_DIR, new_name)
                if not os.path.exists(old_dir):
                    raise FileNotFoundError(f"Scenario directory '{old_dir}' does not exist.")
                if os.path.exists(new_dir):
                    raise FileExistsError(f"A scenario named '{new_name}' already exists.")
                os.rename(old_dir, new_dir)
            except OSError as e:
                logger.error(f"Error renaming scenario directory: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not rename scenario directory.\n{e}")
                return

            # Update the scenario object and save it
            self.current_scenario.name = new_name
            self.current_scenario.save()
            
            # Refresh the UI
            self.load_scenarios()
            self.combo.setCurrentText(new_name)

    def rename_step(self):
        """
        Open a dialog to rename the selected step.
        """
        if not self.current_scenario or self.selected_step_idx is None:
            return

        step = self.current_scenario.steps[self.selected_step_idx]
        old_name = step.get('name', 'Unnamed Step')
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Step", "New name:", text=old_name)

        if ok and new_name and new_name != old_name:
            step['name'] = new_name
            self.current_scenario.save()
            self.refresh_lists()

    def import_scenario(self):
        """
        Open a file dialog to import a scenario from a zip file.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import Scenario', '', 'ZIP Files (*.zip)')
        if path:
            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    # Extract to a temporary directory to inspect
                    temp_dir = os.path.join(CONFIG_DIR, "_temp_import")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    zip_ref.extractall(temp_dir)
                    
                    # Find the scenario name (the directory inside the temp folder)
                    extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                    if not extracted_dirs:
                        raise Exception("No scenario directory found in the zip file.")
                    scenario_name = extracted_dirs[0]
                    
                    # Move the extracted scenario to the main scenarios directory
                    src_path = os.path.join(temp_dir, scenario_name)
                    dest_path = os.path.join(CONFIG_DIR, scenario_name)
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.move(src_path, dest_path)
                    
                    shutil.rmtree(temp_dir)

                logger.info(f'Imported scenario from {path}')
                self.load_scenarios()
                self.combo.setCurrentText(scenario_name)
            except Exception as e:
                logger.error(f'Failed to import scenario: {e}')
                QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to import scenario.\n{e}")

    def export_scenario(self):
        """
        Open a file dialog to export the current scenario to a zip file.
        """
        if not self.current_scenario:
            logger.warning('No scenario selected for export.')
            return

        scenario_dir = self.current_scenario.get_scenario_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Scenario', f'{self.current_scenario.name}.zip', 'ZIP Files (*.zip)')

        if path:
            try:
                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(scenario_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, CONFIG_DIR)
                            zipf.write(file_path, arcname)
                logger.info(f'Exported scenario to {path}')
            except Exception as e:
                logger.error(f'Failed to export scenario: {e}')
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export scenario.\n{e}")

    
    def add_action(self):
        """
        Open the dialog to add a new action to the step.
        """
        """
        Open the dialog to add an action to the current scenario.
        """
        logger.debug('Add Action: Opening action dialog.')
        dlg = ActionDialog(self, self.current_scenario.images)
        if dlg.exec():
            action = dlg.get_action()
            self.current_scenario.actions.append(action)
            self.current_scenario.save()
            self.refresh_lists()
            logger.info(f'Action added: {action}')

    def refresh_lists(self):
        """
        Refresh the step list widget with the current scenario's steps.
        """
        logger.debug('Refreshing steps list.')
        self.steps_list.clear()
        if not self.current_scenario:
            return
        for step in self.current_scenario.steps:
            name = step.get('name', 'Unnamed Step')
            cond = step.get('condition', 'OR')
            imgs = ','.join([img.get('name', 'img') for img in step.get('images', [])])
            acts = ','.join([a.get('type', 'action') for a in step.get('actions', [])])
            self.steps_list.addItem(f"{name} [{cond}] imgs: {imgs} actions: {acts}")

    def refresh_window_list(self, keep_selection=False):
        """
        Populate the window selection dropdown with all open windows and 'Entire Screen'.
        If keep_selection is True, try to keep the current selection.
        """
        self.window_combo.blockSignals(True)
        current = self.window_combo.currentText() if keep_selection else None
        self.window_combo.clear()
        self.window_combo.addItem('Entire Screen')
        try:
            windows = gw.getAllTitles()
            for title in windows:
                if title.strip():
                    self.window_combo.addItem(title)
        except Exception as e:
            logger.error(f'Error listing windows: {e}')
        if keep_selection and current:
            idx = self.window_combo.findText(current)
            if idx != -1:
                self.window_combo.setCurrentIndex(idx)
        self.window_combo.blockSignals(False)

    def save_selected_window(self):
        """
        Save the currently selected window to the scenario's config file.
        """
        if self.current_scenario:
            selected_window = self.window_combo.currentText()
            scenario_dir = self.current_scenario.get_scenario_dir()
            config_path = os.path.join(scenario_dir, 'window_config.json')
            try:
                with open(config_path, 'w') as f:
                    json.dump({'selected_window': selected_window}, f)
            except Exception as e:
                logger.error(f'Failed to save window selection: {e}')

    def load_selected_window_from_config(self):
        """
        Load the selected window from the scenario's config file and update the dropdown.
        """
        if self.current_scenario:
            scenario_dir = self.current_scenario.get_scenario_dir()
            config_path = os.path.join(scenario_dir, 'window_config.json')
            
            # Default to 'Entire Screen'
            win_name = 'Entire Screen'
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                        win_name = data.get('selected_window', 'Entire Screen')
                except Exception as e:
                    logger.error(f'Failed to load window selection: {e}')

            # Use a single shot timer to set the index after the event loop has processed
            def set_window():
                self.refresh_window_list() # Refresh list first
                idx = self.window_combo.findText(win_name)
                if idx != -1:
                    self.window_combo.setCurrentIndex(idx)
                    logger.info(f"UI: Set window to '{win_name}' from config.")
                else:
                    self.window_combo.setCurrentIndex(0)
                    logger.warning(f"UI: Saved window '{win_name}' not found. Defaulting to 'Entire Screen'.")

            QtCore.QTimer.singleShot(100, set_window)
# StepDialog for creating/editing a step
class StepDialog(QtWidgets.QDialog):
    """
    Dialog for creating or editing a step in a scenario.
    Allows editing images, actions, and step properties.
    """
    def __init__(self, parent=None, step=None):
        """
        Initialize the StepDialog for creating or editing a step.
        """
        super().__init__(parent)
        self.setWindowTitle('Step Editor')
        self.resize(600, 400)
        self.images = [] if step is None else step.get('images', [])
        self.actions = [] if step is None else step.get('actions', [])
        self.name_edit = QtWidgets.QLineEdit(step['name'] if step and 'name' in step else '')
        self.cond_combo = QtWidgets.QComboBox()
        self.cond_combo.addItems(['OR', 'AND'])
        if step and 'condition' in step:
            self.cond_combo.setCurrentText(step['condition'])
        # Images
        self.img_list = QtWidgets.QListWidget()
        self.img_list.currentItemChanged.connect(self.update_image_preview)
        for img in self.images:
            self.img_list.addItem(img.get('name', 'img'))
        self.sensitivity_spin = QtWidgets.QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.5, 1.0)
        self.sensitivity_spin.setSingleStep(0.01)
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setValue(0.9)
        self.sensitivity_spin.setToolTip('Sensitivity for selected image (higher = stricter match)')
        self.img_list.currentRowChanged.connect(self.update_sensitivity_spin)
        self.sensitivity_spin.valueChanged.connect(self.save_sensitivity_for_image)
        self.btn_add_img = QtWidgets.QPushButton('Add Image')
        self.btn_add_img.clicked.connect(self.add_image_to_step)
        self.btn_del_img = QtWidgets.QPushButton('Delete Image')
        self.btn_del_img.clicked.connect(self.delete_image)
        self.btn_rename_img = QtWidgets.QPushButton("Rename Image")
        self.btn_rename_img.clicked.connect(self.rename_image)
        self.btn_retake_img = QtWidgets.QPushButton("Retake Screenshot")
        self.btn_retake_img.clicked.connect(self.retake_screenshot)
        self.img_preview = QtWidgets.QLabel('Image Preview')
        self.img_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setMinimumSize(200, 200)
        # Actions
        self.act_list = QtWidgets.QListWidget()
        for act in self.actions:
            self.act_list.addItem(act.get('type', 'action'))
        self.btn_add_act = QtWidgets.QPushButton('Add Action')
        self.btn_add_act.clicked.connect(self.add_action)
        self.btn_del_act = QtWidgets.QPushButton('Delete Action')
        self.btn_del_act.clicked.connect(self.delete_action)
        self.btn_move_up_act = QtWidgets.QPushButton('Move Up')
        self.btn_move_up_act.clicked.connect(self.move_action_up)
        self.btn_move_down_act = QtWidgets.QPushButton('Move Down')
        self.btn_move_down_act.clicked.connect(self.move_action_down)
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Step Name:'), 0, 0)
        layout.addWidget(self.name_edit, 0, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Condition:'), 1, 0)
        layout.addWidget(self.cond_combo, 1, 1, 1, 2)
        
        img_layout = QtWidgets.QHBoxLayout()
        img_list_layout = QtWidgets.QVBoxLayout()
        img_list_layout.addWidget(self.img_list)
        img_list_layout.addWidget(QtWidgets.QLabel('Sensitivity:'))
        img_list_layout.addWidget(self.sensitivity_spin)
        img_list_layout.addWidget(self.btn_add_img)
        img_list_layout.addWidget(self.btn_del_img)
        img_list_layout.addWidget(self.btn_rename_img)
        img_list_layout.addWidget(self.btn_retake_img)
        img_layout.addLayout(img_list_layout)
        img_layout.addWidget(self.img_preview)
        
        layout.addWidget(QtWidgets.QLabel('Images:'), 2, 0)
        layout.addLayout(img_layout, 2, 1, 1, 2)

        layout.addWidget(QtWidgets.QLabel('Actions:'), 3, 0)
        layout.addWidget(self.act_list, 4, 1, 1, 2)
        action_buttons = QtWidgets.QHBoxLayout()
        action_buttons.addWidget(self.btn_add_act)
        action_buttons.addWidget(self.btn_del_act)
        action_buttons.addWidget(self.btn_move_up_act)
        action_buttons.addWidget(self.btn_move_down_act)
        layout.addLayout(action_buttons, 5, 1, 1, 2)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns, 6, 1, 1, 2)
        self.setLayout(layout)

    def move_action_up(self):
        """
        Move the selected action up in the action list.
        """
        idx = self.act_list.currentRow()
        if idx > 0:
            self.actions[idx-1], self.actions[idx] = self.actions[idx], self.actions[idx-1]
            item = self.act_list.takeItem(idx)
            self.act_list.insertItem(idx-1, item)
            self.act_list.setCurrentRow(idx-1)

    def move_action_down(self):
        """
        Move the selected action down in the action list.
        """
        idx = self.act_list.currentRow()
        if idx < len(self.actions) - 1 and idx >= 0:
            self.actions[idx+1], self.actions[idx] = self.actions[idx], self.actions[idx+1]
            item = self.act_list.takeItem(idx)
            self.act_list.insertItem(idx+1, item)
            self.act_list.setCurrentRow(idx+1)

    class NameImageDialog(QtWidgets.QDialog):
        def __init__(self, pixmap, parent=None):
            """
            Initialize the dialog for naming a new image.
            """
            super().__init__(parent)
            self.setWindowTitle("Name Your Image")
            
            # Layouts
            layout = QtWidgets.QVBoxLayout(self)
            form_layout = QtWidgets.QFormLayout()
            
            # Image Preview
            self.preview_label = QtWidgets.QLabel()
            self.preview_label.setPixmap(pixmap.scaled(300, 300, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            layout.addWidget(self.preview_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Name Input
            self.name_edit = QtWidgets.QLineEdit()
            form_layout.addRow("Image Name:", self.name_edit)
            layout.addLayout(form_layout)
            
            # Dialog Buttons
            button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)

        def get_name(self):
            """
            Return the entered image name.
            """
            return self.name_edit.text()


    def add_image_to_step(self):
        """
        Add a new image to the step by taking a screenshot and letting the user select a region.
        """
        main_window = self.parent()
        step_name = self.name_edit.text()
        if not step_name:
            QtWidgets.QMessageBox.warning(self, "Step Name Required", "Please enter a name for the step before adding an image.")
            return

        logger.debug("StepDialog.add_image_to_step: Minimizing all windows for screenshot.")
        all_windows = QtWidgets.QApplication.topLevelWidgets()
        for w in all_windows:
            w.setWindowState(QtCore.Qt.WindowState.WindowMinimized)

        QtWidgets.QApplication.processEvents()
        time.sleep(0.5)

        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if not screen:
                logger.error("StepDialog.add_image_to_step: Could not get primary screen.")
                raise Exception("Could not get primary screen.")
            # Use mss for more reliable screenshot
            import mss.tools
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                qimg = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                full_screenshot_pixmap = QtGui.QPixmap.fromImage(qimg)
            logger.debug("StepDialog.add_image_to_step: Screenshot taken with mss.")
        except Exception as e:
            logger.error(f"StepDialog.add_image_to_step: Screenshot failed: {e}")
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to take screenshot: {e}')
            for w in all_windows:
                w.setWindowState(QtCore.Qt.WindowState.WindowNoState)
                w.showNormal()
                w.activateWindow()
            return
        finally:
            for w in all_windows:
                w.setWindowState(QtCore.Qt.WindowState.WindowNoState)
                w.showNormal()
                w.activateWindow()

        rect_coords = take_screenshot_with_tkinter()
        
        if rect_coords and rect_coords["width"] > 0 and rect_coords["height"] > 0:
            
            # Now, use the coordinates to crop the original full screenshot
            rect = QtCore.QRect(rect_coords["x"], rect_coords["y"], rect_coords["width"], rect_coords["height"])
            cropped_pixmap = full_screenshot_pixmap.copy(rect)

            name_dialog = self.NameImageDialog(cropped_pixmap, self)
            if name_dialog.exec():
                name = name_dialog.get_name()
                scenario_dir = main_window.current_scenario.get_scenario_dir()
                step_images_dir = os.path.join(scenario_dir, "steps", step_name)
                if not os.path.exists(step_images_dir):
                    os.makedirs(step_images_dir)

                safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
                path = os.path.join(step_images_dir, f'{safe_name}.png')

                if cropped_pixmap.save(path, "PNG"):
                    logger.info(f"Screenshot saved to {path}")
                    img_obj = {'path': path, 'region': [rect.x(), rect.y(), rect.width(), rect.height()], 'name': name, 'sensitivity': 0.9}
                    self.images.append(img_obj)
                    self.img_list.addItem(name)
                else:
                    logger.error(f"Failed to save screenshot to {path}")
                    QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to save the screenshot file.')

    def update_image_preview(self, current, previous):
        """
        Update the image preview label when a new image is selected.
        """
        if current:
            idx = self.img_list.row(current)
            path = self.images[idx]['path']
            pixmap = QtGui.QPixmap(path)
            self.img_preview.setPixmap(pixmap.scaled(self.img_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            self.img_preview.setText('Image Preview')

    def update_sensitivity_spin(self, idx):
        """
        Update the sensitivity spinbox for the selected image.
        """
        if idx < 0 or idx >= len(self.images):
            self.sensitivity_spin.setValue(0.9)
            self.sensitivity_spin.setEnabled(False)
            return
        self.sensitivity_spin.setEnabled(True)
        img = self.images[idx]
        self.sensitivity_spin.setValue(img.get('sensitivity', 0.9))

    def save_sensitivity_for_image(self, value):
        """
        Save the sensitivity value for the currently selected image.
        """
        idx = self.img_list.currentRow()
        if idx < 0 or idx >= len(self.images):
            return
        self.images[idx]['sensitivity'] = value

    def delete_image(self):
        """
        Delete the currently selected image from the step.
        """
        idx = self.img_list.currentRow()
        if idx >= 0:
            self.images.pop(idx)
            self.img_list.takeItem(idx)

    def rename_image(self):
        """
        Open a dialog to rename the currently selected image.
        """
        idx = self.img_list.currentRow()
        if idx < 0:
            return

        img_obj = self.images[idx]
        old_name = img_obj.get('name', '')
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Image", "New name:", text=old_name)

        if ok and new_name and new_name != old_name:
            img_obj['name'] = new_name
            self.img_list.item(idx).setText(new_name)

    def retake_screenshot(self):
        """
        Retake the screenshot for the currently selected image.
        """
        idx = self.img_list.currentRow()
        if idx < 0:
            return

        all_windows = QtWidgets.QApplication.topLevelWidgets()
        for w in all_windows:
            w.setWindowState(QtCore.Qt.WindowState.WindowMinimized)

        time.sleep(0.5)

        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if not screen:
                raise Exception("Could not get primary screen.")
            full_screenshot_pixmap = screen.grabWindow(0)
        finally:
            for w in all_windows:
                w.setWindowState(QtCore.Qt.WindowState.WindowNoState)
                w.showNormal()
                w.activateWindow()

        rect_coords = take_screenshot_with_tkinter()

        if rect_coords and rect_coords["width"] > 0 and rect_coords["height"] > 0:
            rect = QtCore.QRect(rect_coords["x"], rect_coords["y"], rect_coords["width"], rect_coords["height"])
            cropped_pixmap = full_screenshot_pixmap.copy(rect)

            img_obj = self.images[idx]
            path = img_obj['path']

            if cropped_pixmap.save(path, "PNG"):
                logger.info(f"Screenshot retaken and saved to {path}")
                img_obj['region'] = [rect.x(), rect.y(), rect.width(), rect.height()]
                self.update_image_preview(self.img_list.currentItem(), None)
            else:
                logger.error(f"Failed to save retaken screenshot to {path}")
                QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to save the retaken screenshot file.')

    def add_action(self):
        dlg = StepActionDialog(self)
        if dlg.exec():
            act = dlg.get_action()
            self.actions.append(act)
            self.act_list.addItem(act.get('type', 'action'))

    def delete_action(self):
        """
        Delete the currently selected action from the step.
        """
        idx = self.act_list.currentRow()
        if idx >= 0:
            self.actions.pop(idx)
            self.act_list.takeItem(idx)

    def get_step(self):
        """
        Return the step as a dictionary for saving.
        """
        return {
            'name': self.name_edit.text(),
            'condition': self.cond_combo.currentText(),
            'images': self.images,
            'actions': self.actions
        }

# StepActionDialog for adding actions to a step
class StepActionDialog(QtWidgets.QDialog):
    """
    Dialog for adding or editing an action for a step.
    Supports click, key, scroll, and delay actions.
    """
    def __init__(self, parent=None):
        """
        Initialize the StepActionDialog for adding or editing a step action.
        """
        super().__init__(parent)
        self.setWindowTitle('Add Action')
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(['click', 'key', 'scroll', 'delay'])
        # Click options
        self.click_btn_combo = QtWidgets.QComboBox()
        self.click_btn_combo.addItems(['left', 'middle', 'right'])
        self.click_pos_combo = QtWidgets.QComboBox()
        self.click_pos_combo.addItems(['center', 'relative', 'absolute'])
        self.rel_x_spin = QtWidgets.QSpinBox()
        self.rel_x_spin.setRange(-10000, 10000)
        self.rel_y_spin = QtWidgets.QSpinBox()
        self.rel_y_spin.setRange(-10000, 10000)
        self.abs_x_spin = QtWidgets.QSpinBox()
        self.abs_x_spin.setRange(0, 10000)
        self.abs_y_spin = QtWidgets.QSpinBox()
        self.abs_y_spin.setRange(0, 10000)
        self.pick_abs_btn = QtWidgets.QPushButton('Pick Position')
        self.pick_abs_btn.clicked.connect(self.pick_absolute_position)
        # Scroll options
        self.scroll_dir_combo = QtWidgets.QComboBox()
        self.scroll_dir_combo.addItems(['up', 'down'])
        self.scroll_amt_spin = QtWidgets.QSpinBox()
        self.scroll_amt_spin.setRange(1, 10000)
        self.scroll_amt_spin.setValue(100)
        # Key options
        self.key_label = QtWidgets.QLabel('(press a key)')
        self.key_value = None
        self.key_capture_btn = QtWidgets.QPushButton('Set Key')
        self.key_capture_btn.clicked.connect(self.capture_key)
        # Delay options
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 60.0)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setValue(1.0)
        # Layout
        self.form = QtWidgets.QFormLayout()
        self.form.addRow('Type:', self.type_combo)
        # Click group
        self.click_group = QtWidgets.QGroupBox('Click Options')
        click_layout = QtWidgets.QFormLayout()
        click_layout.addRow('Button:', self.click_btn_combo)
        click_layout.addRow('Position Mode:', self.click_pos_combo)
        click_layout.addRow('Relative X:', self.rel_x_spin)
        click_layout.addRow('Relative Y:', self.rel_y_spin)
        click_layout.addRow('Absolute X:', self.abs_x_spin)
        click_layout.addRow('Absolute Y:', self.abs_y_spin)
        click_layout.addRow(self.pick_abs_btn)
        self.click_group.setLayout(click_layout)
        self.form.addRow(self.click_group)
        # Scroll group
        self.scroll_group = QtWidgets.QGroupBox('Scroll Options')
        scroll_layout = QtWidgets.QFormLayout()
        scroll_layout.addRow('Direction:', self.scroll_dir_combo)
        scroll_layout.addRow('Amount:', self.scroll_amt_spin)
        self.scroll_group.setLayout(scroll_layout)
        self.form.addRow(self.scroll_group)
        # Key group
        self.key_group = QtWidgets.QGroupBox('Key Options')
        key_layout = QtWidgets.QHBoxLayout()
        key_layout.addWidget(self.key_label)
        key_layout.addWidget(self.key_capture_btn)
        self.key_group.setLayout(key_layout)
        self.form.addRow(self.key_group)
        # Delay group
        self.delay_group = QtWidgets.QGroupBox('Delay Options')
        delay_layout = QtWidgets.QFormLayout()
        delay_layout.addRow('Duration (s):', self.delay_spin)
        self.delay_group.setLayout(delay_layout)
        self.form.addRow(self.delay_group)
        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.form.addWidget(btns)
        self.setLayout(self.form)
        self.type_combo.currentTextChanged.connect(lambda: self.update_fields())
        self.click_pos_combo.currentTextChanged.connect(lambda: self.update_fields())
        self.update_fields()

    def update_fields(self):
        """
        Update the visibility of fields based on the selected action type.
        """
        t = self.type_combo.currentText()
        self.click_group.setVisible(t == 'click')
        self.scroll_group.setVisible(t == 'scroll')
        self.key_group.setVisible(t == 'key')
        self.delay_group.setVisible(t == 'delay')
        # Click position fields
        if t == 'click':
            self.click_group.setVisible(True)
            pos_mode = self.click_pos_combo.currentText()
            # Find the layout rows for rel_x, rel_y, abs_x, abs_y
            for i in range(self.click_group.layout().rowCount()):
                label = self.click_group.layout().itemAt(i, QtWidgets.QFormLayout.ItemRole.LabelRole)
                field = self.click_group.layout().itemAt(i, QtWidgets.QFormLayout.ItemRole.FieldRole)
                if label and field:
                    label_text = label.widget().text() if hasattr(label.widget(), 'text') else ''
                    if label_text == 'Relative X:' or label_text == 'Relative Y:':
                        show = pos_mode == 'relative'
                        label.widget().setVisible(show)
                        field.widget().setVisible(show)
                    elif label_text == 'Absolute X:' or label_text == 'Absolute Y:':
                        show = pos_mode == 'absolute'
                        label.widget().setVisible(show)
                        field.widget().setVisible(show)
            self.pick_abs_btn.setVisible(pos_mode == 'absolute')
        else:
            self.click_group.setVisible(False)

    def pick_absolute_position(self):
        """
        Open an overlay to pick an absolute position on the screen.
        """
        # Overlay to pick a position on screen
        class PosOverlay(QtWidgets.QWidget):
            pos_picked = QtCore.pyqtSignal(int, int)
            def __init__(self):
                super().__init__()
                self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
                self.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            def mousePressEvent(self, event):
                self.pos_picked.emit(event.globalX(), event.globalY())
                self.close()
            def paintEvent(self, event):
                pass
        overlay = PosOverlay()
        overlay.pos_picked.connect(self.set_absolute_position)
        overlay.showFullScreen()

    def set_absolute_position(self, x, y):
        """
        Set the absolute position fields to the picked coordinates.
        """
        self.abs_x_spin.setValue(x)
        self.abs_y_spin.setValue(y)

    def capture_key(self):
        """
        Open a dialog to capture a key press from the user.
        """
        # Open a modal dialog to capture a key press
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Press a key')
        label = QtWidgets.QLabel('Press any key...', dlg)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        dlg.setLayout(layout)
        key_captured = {}
        def on_key(event):
            if event.type() == QtCore.QEvent.Type.KeyPress:
                key_captured['key'] = event.key()
                key_captured['text'] = event.text()
                dlg.accept()
                return True
            return False
        dlg.installEventFilter(self)
        orig_event = dlg.event
        def event_filter(obj, event):
            if event.type() == QtCore.QEvent.Type.KeyPress:
                key = event.key()
                text = event.text()
                self.key_value = text if text else str(key)
                self.key_label.setText(f'Key: {self.key_value}')
                dlg.accept()
                return True
            return orig_event(event)
        dlg.event = event_filter
        dlg.exec()

    def get_action(self):
        """
        Return the action as a dictionary for saving.
        """
        t = self.type_combo.currentText()
        params = {}
        if t == 'click':
            params['button'] = self.click_btn_combo.currentText()
            pos_mode = self.click_pos_combo.currentText()
            params['pos_type'] = pos_mode
            if pos_mode == 'relative':
                params['rel_x'] = self.rel_x_spin.value()
                params['rel_y'] = self.rel_y_spin.value()
            elif pos_mode == 'absolute':
                params['abs_x'] = self.abs_x_spin.value()
                params['abs_y'] = self.abs_y_spin.value()
        elif t == 'scroll':
            params['direction'] = self.scroll_dir_combo.currentText()
            params['amount'] = self.scroll_amt_spin.value()
        elif t == 'key':
            params['key'] = self.key_value if self.key_value else ''
        elif t == 'delay':
            params['duration'] = self.delay_spin.value()
        return {'type': t, 'params': params}

    
    def perform_step_action(self, action, loc, shape):
        """
        Perform the given step action at the specified location.
        """
        act_type = action['type']
        params = action['params']
        logger.debug(f'Performing step action: {act_type}, params={params}, loc={loc}, shape={shape}')
        try:
            if act_type == 'click':
                pos_type = params.get('pos_type', 'center')
                if pos_type == 'center':
                    x = loc[0] + shape[1] // 2
                    y = loc[1] + shape[0] // 2
                elif pos_type == 'relative':
                    x = loc[0] + params.get('rel_x', 0)
                    y = loc[1] + params.get('rel_y', 0)
                elif pos_type == 'absolute':
                    x = params.get('abs_x', 0)
                    y = params.get('abs_y', 0)
                else:
                    x, y = loc[0], loc[1]
                button = params.get('button', 'left')
                logger.info(f'Clicking at ({x}, {y}) with button {button}')
                pyautogui.click(x, y, button=button)
            elif act_type == 'key':
                key = params.get('key', 'enter')
                logger.info(f'Pressing key: {key}')
                pyautogui.press(key)
            elif act_type == 'scroll':
                amount = params.get('amount', 0)
                direction = params.get('direction', 'up')
                logger.info(f'Scrolling: {amount} direction: {direction}')
                if direction == 'up':
                    pyautogui.scroll(amount)
                else:
                    pyautogui.scroll(-abs(amount))
            elif act_type == 'delay':
                duration = params.get('duration', 1)
                logger.info(f'Delay for {duration} seconds')
                time.sleep(duration)
        except Exception as e:
            logger.error(f'Error performing step action {act_type}: {e}')

class ActionDialog(QtWidgets.QDialog):
    """
    Dialog for adding a generic action to a scenario (legacy/unused in main flow).
    """
    def __init__(self, parent=None, images=None):
        """
        Initialize the ActionDialog for adding a generic action.
        """
        super().__init__(parent)
        self.setWindowTitle('Add Action')
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(['click', 'key', 'scroll'])
        self.param_edit = QtWidgets.QLineEdit()
        self.delay_before = QtWidgets.QSpinBox()
        self.delay_after = QtWidgets.QSpinBox()
        self.delay_before.setMaximum(10000)
        self.delay_after.setMaximum(10000)
        self.images = images or []
        self.image_checks = []
        self.logic_combo = QtWidgets.QComboBox()
        self.logic_combo.addItems(['OR', 'AND'])
        layout = QtWidgets.QFormLayout()
        layout.addRow('Type:', self.type_combo)
        layout.addRow('Params (JSON):', self.param_edit)
        layout.addRow('Delay Before (s):', self.delay_before)
        layout.addRow('Delay After (s):', self.delay_after)
        layout.addRow('Image Logic:', self.logic_combo)
        img_group = QtWidgets.QGroupBox('Trigger Images')
        img_layout = QtWidgets.QVBoxLayout()
        for img in self.images:
            cb = QtWidgets.QCheckBox(img.get('name', os.path.basename(img['path'])))
            img_layout.addWidget(cb)
            self.image_checks.append(cb)
        img_group.setLayout(img_layout)
        layout.addRow(img_group)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setLayout(layout)

    def get_action(self):
        """
        Return the action as a dictionary for saving.
        """
        try:
            params = json.loads(self.param_edit.text())
        except Exception:
            params = {}
        params['delay_before'] = self.delay_before.value()
        params['delay_after'] = self.delay_after.value()
        image_refs = [img.get('name', os.path.basename(img['path'])) for img, cb in zip(self.images, self.image_checks) if cb.isChecked()]
        logic = self.logic_combo.currentText()
        return {'type': self.type_combo.currentText(), 'params': params, 'image_refs': image_refs, 'logic': logic}

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
