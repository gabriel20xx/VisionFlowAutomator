
import sys
import os
import json
import threading
import time
import logging
from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
import pyautogui
import mss
from pynput import keyboard

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('scenario_automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_DIR = 'scenarios'
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

class Scenario:
    def __init__(self, name, steps=None):
        self.name = name
        self.steps = steps or []  # List of Step dicts

    def to_dict(self):
        return {'name': self.name, 'steps': self.steps}

    @staticmethod
    def from_dict(data):
        return Scenario(data['name'], data.get('steps', []))

    def save(self):
        try:
            with open(os.path.join(CONFIG_DIR, f'{self.name}.json'), 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Scenario '{self.name}' saved.")
        except Exception as e:
            logger.error(f"Failed to save scenario '{self.name}': {e}")

    @staticmethod
    def load(name):
        try:
            with open(os.path.join(CONFIG_DIR, f'{name}.json'), 'r') as f:
                scenario = Scenario.from_dict(json.load(f))
            logger.info(f"Scenario '{name}' loaded.")
            return scenario
        except Exception as e:
            logger.error(f"Failed to load scenario '{name}': {e}")
            return None

    @staticmethod
    def list_all():
        return [f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]

class ScreenshotOverlay(QtWidgets.QWidget):
    region_selected = QtCore.pyqtSignal(QtCore.QRect)
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.Tool
        )
        self.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.start = None
        self.end = None
        self.selection_rect = None
        self._bg = self._grab_bg()

    def _grab_bg(self):
        # Grab the current screen as a QPixmap for dimming
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            return screen.grabWindow(0)
        return None

    def mousePressEvent(self, event):
        logger.debug('ScreenshotOverlay: Mouse press event.')
        self.start = event.pos()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, event):
        logger.debug('ScreenshotOverlay: Mouse move event.')
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        logger.debug('ScreenshotOverlay: Mouse release event.')
        self.end = event.pos()
        self.selection_rect = QtCore.QRect(self.start, self.end).normalized()
        logger.info(f'ScreenshotOverlay: Region selected: {self.selection_rect}')
        self.region_selected.emit(self.selection_rect)
        self.close()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        if self._bg:
            # Draw the background screenshot
            qp.drawPixmap(self.rect(), self._bg)
            # Draw the semi-transparent dimming overlay
            qp.setBrush(QtGui.QColor(0, 0, 0, 120))
            qp.drawRect(self.rect())

            # If a selection is being made, clear the dimmed area and draw a border
            if self.start and self.end:
                selection = QtCore.QRect(self.start, self.end).normalized()
                # Redraw the original screenshot portion within the selection
                qp.drawPixmap(selection, self._bg, selection)
                # Draw the selection border
                qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
                qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                qp.drawRect(selection)

class MainWindow(QtWidgets.QMainWindow):
    def automation_loop(self):
        logger.info('Automation loop started.')
        try:
            while self.running:
                screen = pyautogui.screenshot()
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
                            logger.debug(f"Detection for {img['name']}: max_val={max_val}")
                            if max_val > 0.9:
                                detections[img['name']] = (max_loc, template.shape)
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
                    logger.debug(f"Step '{step.get('name', 'step')}' trigger check: found={found}, cond={cond}, trigger={trigger}")
                    if trigger:
                        # Use the first detected image for position
                        ref_img = found[0] if found else step.get('images', [])[0]
                        loc, shape = detections.get(ref_img.get('name'), ((0, 0), (0, 0, 0)))
                        for act in step.get('actions', []):
                            self.perform_step_action(act, loc, shape)
                        logger.info(f"Performed actions for step: {step.get('name', 'step')}")
                        time.sleep(1)  # Prevent spamming
                time.sleep(0.2)
        except Exception as e:
            logger.error(f'Automation loop error: {e}')
    def start_automation(self):
        logger.info('Starting automation.')
        if not self.current_scenario or self.running:
            logger.warning('Start Automation: No scenario selected or already running.')
            return
        self.running = True
        self.worker = threading.Thread(target=self.automation_loop, daemon=True)
        self.worker.start()
        self.listener = keyboard.GlobalHotKeys({self.hotkey: self.stop_automation})
        self.listener.start()

    def stop_automation(self):
        logger.info('Stopping automation.')
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Scenario Image Automation')
        self.setGeometry(100, 100, 1000, 700)
        self.running = False
        self.hotkey = '<ctrl>+<alt>+s'
        self.listener = None
        self.worker = None
        self.current_scenario = None
        self.selected_step_idx = None
        self.init_ui()
        self.load_scenarios()

    def init_ui(self):
        # Scenario selection
        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self.select_scenario)
        self.btn_new = QtWidgets.QPushButton('New Scenario')
        self.btn_new.clicked.connect(self.create_scenario)
        self.btn_import = QtWidgets.QPushButton('Import')
        self.btn_import.clicked.connect(self.import_scenario)
        self.btn_export = QtWidgets.QPushButton('Export')
        self.btn_export.clicked.connect(self.export_scenario)
        # Steps list
        self.steps_list = QtWidgets.QListWidget()
        self.steps_list.currentRowChanged.connect(self.select_step)
        self.btn_add_step = QtWidgets.QPushButton('Add Step')
        self.btn_add_step.clicked.connect(self.add_step)
        self.btn_edit_step = QtWidgets.QPushButton('Edit Step')
        self.btn_edit_step.clicked.connect(self.edit_step)
        self.btn_del_step = QtWidgets.QPushButton('Delete Step')
        self.btn_del_step.clicked.connect(self.delete_step)
        # Start/Stop
        self.btn_start = QtWidgets.QPushButton('Start')
        self.btn_start.clicked.connect(self.start_automation)
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.btn_stop.clicked.connect(self.stop_automation)
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Scenario:'), 0, 0)
        layout.addWidget(self.combo, 0, 1)
        layout.addWidget(self.btn_new, 0, 2)
        layout.addWidget(self.btn_import, 0, 3)
        layout.addWidget(self.btn_export, 0, 4)
        layout.addWidget(QtWidgets.QLabel('Steps:'), 1, 0)
        layout.addWidget(self.steps_list, 1, 1, 4, 4)
        layout.addWidget(self.btn_add_step, 5, 1)
        layout.addWidget(self.btn_edit_step, 5, 2)
        layout.addWidget(self.btn_del_step, 5, 3)
        layout.addWidget(self.btn_start, 6, 1)
        layout.addWidget(self.btn_stop, 6, 2)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def select_step(self, idx):
        self.selected_step_idx = idx

    def add_step(self):
        dlg = StepDialog(self)
        if dlg.exec():
            step = dlg.get_step()
            self.current_scenario.steps.append(step)
            self.current_scenario.save()
            self.refresh_lists()
            logger.info(f'Added step: {step}')

    def edit_step(self):
        idx = self.selected_step_idx
        if idx is None or idx < 0 or idx >= len(self.current_scenario.steps):
            return
        step = self.current_scenario.steps[idx]
        dlg = StepDialog(self, step)
        if dlg.exec():
            self.current_scenario.steps[idx] = dlg.get_step()
            self.current_scenario.save()
            self.refresh_lists()
            logger.info(f'Edited step at idx {idx}')

    def delete_step(self):
        idx = self.selected_step_idx
        if idx is None or idx < 0 or idx >= len(self.current_scenario.steps):
            return
        del self.current_scenario.steps[idx]
        self.current_scenario.save()
        self.refresh_lists()
        logger.info(f'Deleted step at idx {idx}')

    def load_scenarios(self):
        logger.debug('Loading scenarios...')
        self.combo.clear()
        for name in Scenario.list_all():
            self.combo.addItem(name)
        if self.combo.count() > 0:
            self.combo.setCurrentIndex(0)
        self.refresh_lists()

    def select_scenario(self):
        name = self.combo.currentText()
        logger.info(f'Scenario selected: {name}')
        if name:
            self.current_scenario = Scenario.load(name)
            self.refresh_lists()

    def create_scenario(self):
        name, ok = QtWidgets.QInputDialog.getText(self, 'New Scenario', 'Enter scenario name:')
        if ok and name:
            logger.info(f'Creating new scenario: {name}')
            s = Scenario(name)
            s.save()
            self.load_scenarios()
            self.combo.setCurrentText(name)

    def import_scenario(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import Scenario', '', 'JSON Files (*.json)')
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                s = Scenario.from_dict(data)
                s.save()
                logger.info(f'Imported scenario from {path}')
                self.load_scenarios()
                self.combo.setCurrentText(s.name)
            except Exception as e:
                logger.error(f'Failed to import scenario: {e}')

    def export_scenario(self):
        if not self.current_scenario:
            logger.warning('No scenario selected for export.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Scenario', f'{self.current_scenario.name}.json', 'JSON Files (*.json)')
        if path:
            try:
                with open(path, 'w') as f:
                    json.dump(self.current_scenario.to_dict(), f, indent=2)
                logger.info(f'Exported scenario to {path}')
            except Exception as e:
                logger.error(f'Failed to export scenario: {e}')

    def add_image(self):
        if not self.current_scenario:
            logger.warning('Add Image: No scenario selected.')
            QtWidgets.QMessageBox.warning(self, 'No Scenario Selected', 'Please select or create a scenario first.')
            return
        try:
            logger.debug('[DEBUG] add_image called, preparing to show ScreenshotOverlay')
            overlay = ScreenshotOverlay()
            logger.debug('[DEBUG] Connecting region_selected signal to save_screenshot')
            overlay.region_selected.connect(lambda rect: self.save_screenshot(rect))
            logger.debug('[DEBUG] Showing ScreenshotOverlay full screen')
            overlay.showFullScreen()
        except Exception as e:
            logger.error(f'Add Image: Could not start screenshot overlay: {e}')
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not start screenshot overlay.\n{e}')

    def save_screenshot(self, rect):
        logger.debug(f'[DEBUG] Entered save_screenshot with rect={rect}')
        try:
            # Validate region
            logger.debug('[DEBUG] Validating region')
            if rect.width() <= 0 or rect.height() <= 0:
                QtWidgets.QMessageBox.critical(self, 'Screenshot Error', 'Selected region is invalid (zero width or height).')
                logger.error('[DEBUG] Save Screenshot: Invalid region (zero width or height).')
                return
            screen = QtWidgets.QApplication.primaryScreen()
            screen_geo = screen.geometry() if screen else None
            logger.debug(f'[DEBUG] Screen geometry: {screen_geo}')
            if screen_geo and (
                rect.x() < 0 or rect.y() < 0 or
                rect.x() + rect.width() > screen_geo.width() or
                rect.y() + rect.height() > screen_geo.height()
            ):
                QtWidgets.QMessageBox.critical(self, 'Screenshot Error', 'Selected region is out of screen bounds.')
                logger.error('[DEBUG] Save Screenshot: Region out of screen bounds.')
                return
            logger.debug('[DEBUG] Region validated, proceeding to mss')
            with mss.mss() as sct:
                monitor = {
                    "top": rect.y(),
                    "left": rect.x(),
                    "width": rect.width(),
                    "height": rect.height()
                }
                logger.debug(f'[DEBUG] Save Screenshot: monitor={monitor}')
                try:
                    logger.debug('[DEBUG] Calling sct.grab')
                    sct_img = sct.grab(monitor)
                    logger.debug('[DEBUG] sct.grab succeeded')
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Screenshot Error', f'Failed to grab screenshot with mss.\n{e}')
                    logger.error(f'[DEBUG] Save Screenshot: mss.grab failed: {e}')
                    return
                img = np.array(sct_img)
                logger.debug('[DEBUG] Converted sct_img to numpy array')
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                logger.debug('[DEBUG] Converted image to BGR')
                name, ok = QtWidgets.QInputDialog.getText(self, 'Image Name', 'Enter image name:')
                if ok and name:
                    path = os.path.join(CONFIG_DIR, f'{self.current_scenario.name}_{name}.png')
                    cv2.imwrite(path, img)
                    logger.debug(f'[DEBUG] Image written to {path}')
                    self.current_scenario.images.append({'path': path, 'region': [rect.x(), rect.y(), rect.width(), rect.height()], 'name': name})
                    self.current_scenario.save()
                    self.refresh_lists()
                    logger.info(f'Screenshot saved: {path}')
                else:
                    logger.info('[DEBUG] Save Screenshot: Cancelled or no name entered.')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Screenshot Error', f'Failed to capture or save screenshot.\n{e}')
            logger.error(f'[DEBUG] Save Screenshot: Failed to capture or save screenshot: {e}')

    def add_action(self):
        logger.debug('Add Action: Opening action dialog.')
        dlg = ActionDialog(self, self.current_scenario.images)
        if dlg.exec():
            action = dlg.get_action()
            self.current_scenario.actions.append(action)
            self.current_scenario.save()
            self.refresh_lists()
            logger.info(f'Action added: {action}')

    def refresh_lists(self):
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
# StepDialog for creating/editing a step
class StepDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, step=None):
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
        for img in self.images:
            self.img_list.addItem(img.get('name', 'img'))
        self.btn_add_img = QtWidgets.QPushButton('Add Image')
        self.btn_add_img.clicked.connect(self.add_image_to_step)
        self.btn_del_img = QtWidgets.QPushButton('Delete Image')
        self.btn_del_img.clicked.connect(self.delete_image)
        # Actions
        self.act_list = QtWidgets.QListWidget()
        for act in self.actions:
            self.act_list.addItem(act.get('type', 'action'))
        self.btn_add_act = QtWidgets.QPushButton('Add Action')
        self.btn_add_act.clicked.connect(self.add_action)
        self.btn_del_act = QtWidgets.QPushButton('Delete Action')
        self.btn_del_act.clicked.connect(self.delete_action)
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Step Name:'), 0, 0)
        layout.addWidget(self.name_edit, 0, 1)
        layout.addWidget(QtWidgets.QLabel('Condition:'), 1, 0)
        layout.addWidget(self.cond_combo, 1, 1)
        layout.addWidget(QtWidgets.QLabel('Images:'), 2, 0)
        layout.addWidget(self.img_list, 2, 1, 1, 2)
        layout.addWidget(self.btn_add_img, 3, 1)
        layout.addWidget(self.btn_del_img, 3, 2)
        layout.addWidget(QtWidgets.QLabel('Actions:'), 4, 0)
        layout.addWidget(self.act_list, 4, 1, 1, 2)
        layout.addWidget(self.btn_add_act, 5, 1)
        layout.addWidget(self.btn_del_act, 5, 2)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns, 6, 1, 1, 2)
        self.setLayout(layout)

    def add_image_to_step(self):
        try:
            self.overlay = ScreenshotOverlay()
            self.overlay.region_selected.connect(self.save_screenshot)
            self.overlay.showFullScreen()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not start screenshot overlay.\n{e}')

    def save_screenshot(self, rect):
        with mss.mss() as sct:
            monitor = {
                "top": rect.y(),
                "left": rect.x(),
                "width": rect.width(),
                "height": rect.height()
            }
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            name, ok = QtWidgets.QInputDialog.getText(self, 'Image Name', 'Enter image name:')
            if ok and name:
                path = os.path.join(CONFIG_DIR, f'step_{name}.png')
                cv2.imwrite(path, img)
                img_obj = {'path': path, 'region': [rect.x(), rect.y(), rect.width(), rect.height()], 'name': name}
                self.images.append(img_obj)
                self.img_list.addItem(name)

    def delete_image(self):
        idx = self.img_list.currentRow()
        if idx >= 0:
            self.images.pop(idx)
            self.img_list.takeItem(idx)

    def add_action(self):
        dlg = StepActionDialog(self)
        if dlg.exec():
            act = dlg.get_action()
            self.actions.append(act)
            self.act_list.addItem(act.get('type', 'action'))

    def delete_action(self):
        idx = self.act_list.currentRow()
        if idx >= 0:
            self.actions.pop(idx)
            self.act_list.takeItem(idx)

    def get_step(self):
        return {
            'name': self.name_edit.text(),
            'condition': self.cond_combo.currentText(),
            'images': self.images,
            'actions': self.actions
        }

# StepActionDialog for adding actions to a step
class StepActionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
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
        self.abs_x_spin.setValue(x)
        self.abs_y_spin.setValue(y)

    def capture_key(self):
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
    def __init__(self, parent=None, images=None):
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
