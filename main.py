
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

class CropDialog(QtWidgets.QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Area")
        self.pixmap = pixmap

        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(self.pixmap)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        
        self.rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
        self.start_pos = None
        self.selection_rect = None

    def mousePressEvent(self, event):
        self.start_pos = event.position().toPoint()
        self.rubber_band.setGeometry(QtCore.QRect(self.start_pos, QtCore.QSize()))
        self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.rubber_band.setGeometry(QtCore.QRect(self.start_pos, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if self.start_pos:
            self.selection_rect = self.rubber_band.geometry()
            self.accept()

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
        self.btn_rename_scenario = QtWidgets.QPushButton("Rename Scenario")
        self.btn_rename_scenario.clicked.connect(self.rename_scenario)
        self.btn_rename_step = QtWidgets.QPushButton("Rename Step")
        self.btn_rename_step.clicked.connect(self.rename_step)
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
        layout.addWidget(self.btn_rename_scenario, 0, 5)
        layout.addWidget(QtWidgets.QLabel('Steps:'), 1, 0)
        layout.addWidget(self.steps_list, 1, 1, 4, 4)
        layout.addWidget(self.btn_add_step, 5, 1)
        layout.addWidget(self.btn_edit_step, 5, 2)
        layout.addWidget(self.btn_del_step, 5, 3)
        layout.addWidget(self.btn_rename_step, 5, 4)
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

    def rename_scenario(self):
        if not self.current_scenario:
            return
        
        old_name = self.current_scenario.name
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Scenario", "New name:", text=old_name)
        
        if ok and new_name and new_name != old_name:
            # Rename the scenario file
            try:
                os.rename(os.path.join(CONFIG_DIR, f"{old_name}.json"), os.path.join(CONFIG_DIR, f"{new_name}.json"))
            except OSError as e:
                logger.error(f"Error renaming scenario file: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not rename scenario file.\n{e}")
                return

            # Update the scenario object and save it
            self.current_scenario.name = new_name
            self.current_scenario.save()
            
            # Refresh the UI
            self.load_scenarios()
            self.combo.setCurrentText(new_name)

    def rename_step(self):
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
        self.img_list.currentItemChanged.connect(self.update_image_preview)
        for img in self.images:
            self.img_list.addItem(img.get('name', 'img'))
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
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Step Name:'), 0, 0)
        layout.addWidget(self.name_edit, 0, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Condition:'), 1, 0)
        layout.addWidget(self.cond_combo, 1, 1, 1, 2)
        
        img_layout = QtWidgets.QHBoxLayout()
        img_list_layout = QtWidgets.QVBoxLayout()
        img_list_layout.addWidget(self.img_list)
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
        layout.addLayout(action_buttons, 5, 1, 1, 2)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns, 6, 1, 1, 2)
        self.setLayout(layout)

    class NameImageDialog(QtWidgets.QDialog):
    def __init__(self, pixmap, parent=None):
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
        return self.name_edit.text()


    def add_image_to_step(self):
        main_window = self.parent()
        main_window.hide()
        # A brief delay to ensure the window is hidden before taking a screenshot
        time.sleep(0.3)
        
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if not screen:
                raise Exception("Could not get primary screen.")
            
            full_screenshot_pixmap = screen.grabWindow(0)
        finally:
            main_window.show()

        crop_dialog = CropDialog(full_screenshot_pixmap, self)
        if crop_dialog.exec():
            rect = crop_dialog.selection_rect
            if rect and not rect.isEmpty():
                cropped_pixmap = full_screenshot_pixmap.copy(rect)
                
                name_dialog = NameImageDialog(cropped_pixmap, self)
                if name_dialog.exec():
                    name = name_dialog.get_name()
                    # Sanitize name to be a valid filename
                    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
                    path = os.path.join(CONFIG_DIR, f'step_{safe_name}.png')
                    
                    if cropped_pixmap.save(path, "PNG"):
                        logger.info(f"Screenshot saved to {path}")
                        img_obj = {'path': path, 'region': [rect.x(), rect.y(), rect.width(), rect.height()], 'name': name}
                        self.images.append(img_obj)
                        self.img_list.addItem(name)
                    else:
                        logger.error(f"Failed to save screenshot to {path}")
                        QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to save the screenshot file.')

    def update_image_preview(self, current, previous):
        if current:
            idx = self.img_list.row(current)
            path = self.images[idx]['path']
            pixmap = QtGui.QPixmap(path)
            self.img_preview.setPixmap(pixmap.scaled(self.img_preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            self.img_preview.setText('Image Preview')

    def delete_image(self):
        idx = self.img_list.currentRow()
        if idx >= 0:
            self.images.pop(idx)
            self.img_list.takeItem(idx)

    def rename_image(self):
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
        idx = self.img_list.currentRow()
        if idx < 0:
            return

        main_window = self.parent()
        self.hide()
        main_window.hide()
        time.sleep(0.3)

        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if not screen:
                raise Exception("Could not get primary screen.")
            full_screenshot_pixmap = screen.grabWindow(0)
        finally:
            main_window.show()
            self.show()

        crop_dialog = CropDialog(full_screenshot_pixmap, self)
        if crop_dialog.exec():
            rect = crop_dialog.selection_rect
            if rect and not rect.isEmpty():
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
