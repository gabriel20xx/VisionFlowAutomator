import sys
import os
import json
import threading
import time
import logging
import zipfile
import shutil
import gc
import weakref
from functools import lru_cache
from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import pygetwindow as gw
import numpy as np
import pyautogui
from pynput import keyboard
import tkinter as tk
from PIL import ImageGrab, Image, ImageTk

# Disable the PyAutoGUI fail-safe feature and optimize settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01  # Reduce pause between actions
pyautogui.MINIMUM_DURATION = 0  # Remove minimum duration for faster execution

# Setup logs directory
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Setup logging with optimized settings
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to reduce log volume
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'scenario_automation.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Create a more efficient logger for performance-critical sections
perf_logger = logging.getLogger('performance')
perf_logger.setLevel(logging.WARNING)  # Only log warnings and errors for performance

logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """
    Get GPU memory information using multiple methods.
    Supports both discrete and integrated GPUs across Windows, Linux, and macOS.
    Returns dict with GPU memory info or basic info if no GPU found.
    Uses caching to prevent repeated expensive system calls.
    """
    # Return cached info immediately if available and recent
    current_time = time.time()
    cache_duration = 30  # seconds
    
    if (hasattr(get_gpu_memory_info, '_cached_info') and 
        hasattr(get_gpu_memory_info, '_cache_time') and
        current_time - get_gpu_memory_info._cache_time < cache_duration):
        return get_gpu_memory_info._cached_info
    
    # If we're already updating in background, return cached or default
    if hasattr(get_gpu_memory_info, '_updating') and get_gpu_memory_info._updating:
        return getattr(get_gpu_memory_info, '_cached_info', 
                      {'has_gpu': False, 'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
                       'utilization_percent': 0, 'gpu_name': 'Loading...', 'method': 'loading'})
    
    # Start background update
    get_gpu_memory_info._updating = True
    
    def update_gpu_info():
        """Background thread function to update GPU info."""
        import json
        import subprocess
        import platform
        import os
        
        gpu_info = {'has_gpu': False, 'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization_percent': 0, 'gpu_name': 'Unknown'}
        
        try:
            # Try nvidia-ml-py (NVIDIA GPUs - most detailed info)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = mem_info.total / 1024 / 1024
                used_mb = mem_info.used / 1024 / 1024
                free_mb = mem_info.free / 1024 / 1024
                utilization_percent = (used_mb / total_mb) * 100
                
                # Get GPU name
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                gpu_info.update({
                    'has_gpu': True,
                    'total_mb': total_mb,
                    'used_mb': used_mb,
                    'free_mb': free_mb,
                    'utilization_percent': utilization_percent,
                    'gpu_name': gpu_name,
                    'method': 'pynvml'
                })
                
                # Cache the result and mark as not updating
                get_gpu_memory_info._cached_info = gpu_info
                get_gpu_memory_info._cache_time = time.time()
                get_gpu_memory_info._updating = False
                return
                
            except (ImportError, Exception):
                pass
            
            # Try GPUtil (NVIDIA GPUs alternative)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    total_mb = gpu.memoryTotal
                    used_mb = gpu.memoryUsed
                    free_mb = gpu.memoryFree
                    utilization_percent = (used_mb / total_mb) * 100
                    
                    gpu_info.update({
                        'has_gpu': True,
                        'total_mb': total_mb,
                        'used_mb': used_mb,
                        'free_mb': free_mb,
                        'utilization_percent': utilization_percent,
                        'gpu_name': gpu.name,
                        'method': 'GPUtil'
                    })
                    
                    # Cache the result and mark as not updating
                    get_gpu_memory_info._cached_info = gpu_info
                    get_gpu_memory_info._cache_time = time.time()
                    get_gpu_memory_info._updating = False
                    return
                    
            except (ImportError, Exception):
                pass
            
            # Try Windows-specific methods with very short timeouts
            try:
                # Try nvidia-smi command for NVIDIA GPUs
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=1)  # Very short timeout
                    
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            parts = lines[0].split(', ')
                            if len(parts) >= 4:
                                gpu_name = parts[0].strip()
                                total_mb = float(parts[1].strip())
                                used_mb = float(parts[2].strip())
                                free_mb = float(parts[3].strip())
                                utilization_percent = (used_mb / total_mb) * 100
                                
                                gpu_info.update({
                                    'has_gpu': True,
                                    'total_mb': total_mb,
                                    'used_mb': used_mb,
                                    'free_mb': free_mb,
                                    'utilization_percent': utilization_percent,
                                    'gpu_name': gpu_name,
                                    'method': 'nvidia-smi'
                                })
                                
                                # Cache the result and mark as not updating
                                get_gpu_memory_info._cached_info = gpu_info
                                get_gpu_memory_info._cache_time = time.time()
                                get_gpu_memory_info._updating = False
                                return
                                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    pass
                
                # Skip expensive WMI calls for now to prevent hanging
                # These can be re-enabled later if needed, but with proper threading
                
            except Exception:
                pass
            
        except Exception:
            pass
        finally:
            # Always mark as not updating and cache result
            get_gpu_memory_info._cached_info = gpu_info
            get_gpu_memory_info._cache_time = time.time()
            get_gpu_memory_info._updating = False
    
    # Start background thread for GPU detection
    import threading
    gpu_thread = threading.Thread(target=update_gpu_info, daemon=True)
    gpu_thread.start()
    
    # Return cached info if available, otherwise return loading state
    return getattr(get_gpu_memory_info, '_cached_info', 
                  {'has_gpu': False, 'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 
                   'utilization_percent': 0, 'gpu_name': 'Loading...', 'method': 'loading'})
    
    gpu_info = {'has_gpu': False, 'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization_percent': 0, 'gpu_name': 'Unknown'}
    
    # Try nvidia-ml-py (NVIDIA GPUs - most detailed info)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        
        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = mem_info.total / 1024 / 1024
        used_mb = mem_info.used / 1024 / 1024
        free_mb = mem_info.free / 1024 / 1024
        utilization_percent = (used_mb / total_mb) * 100
        
        # Get GPU name
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        
        gpu_info.update({
            'has_gpu': True,
            'total_mb': total_mb,
            'used_mb': used_mb,
            'free_mb': free_mb,
            'utilization_percent': utilization_percent,
            'gpu_name': gpu_name,
            'method': 'pynvml'
        })
        logger.debug(f"GPU detected via pynvml: {gpu_name}, {total_mb:.0f}MB total")
        
        # Cache the result
        get_gpu_memory_info._cached_info = gpu_info
        get_gpu_memory_info._cache_time = current_time
        return gpu_info
        
    except (ImportError, Exception) as e:
        logger.debug(f"pynvml not available or failed: {e}")
    
    # Try GPUtil (NVIDIA GPUs alternative)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            total_mb = gpu.memoryTotal
            used_mb = gpu.memoryUsed
            free_mb = gpu.memoryFree
            utilization_percent = (used_mb / total_mb) * 100
            
            gpu_info.update({
                'has_gpu': True,
                'total_mb': total_mb,
                'used_mb': used_mb,
                'free_mb': free_mb,
                'utilization_percent': utilization_percent,
                'gpu_name': gpu.name,
                'method': 'GPUtil'
            })
            logger.debug(f"GPU detected via GPUtil: {gpu.name}, {total_mb:.0f}MB total")
            
            # Cache the result
            get_gpu_memory_info._cached_info = gpu_info
            get_gpu_memory_info._cache_time = current_time
            return gpu_info
            
    except (ImportError, Exception) as e:
        logger.debug(f"GPUtil not available or failed: {e}")
    
    # Try psutil for basic GPU info (limited)
    try:
        import psutil
        # Check if there are any GPU-related processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info['name'].lower()
                if any(gpu_process in proc_name for gpu_process in ['nvidia', 'amd', 'intel', 'gpu']):
                    gpu_info.update({
                        'has_gpu': True,
                        'total_mb': 0,  # Can't get detailed info with psutil
                        'used_mb': 0,
                        'free_mb': 0,
                        'utilization_percent': 0,
                        'gpu_name': 'Detected via process',
                        'method': 'psutil_detection'
                    })
                    logger.debug(f"GPU detected via psutil process detection: {proc_name}")
                    
                    # Cache the result
                    get_gpu_memory_info._cached_info = gpu_info
                    get_gpu_memory_info._cache_time = current_time
                    return gpu_info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
    except ImportError:
        pass
    
    # Try Windows-specific methods
    try:
        # Try nvidia-smi command for NVIDIA GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)  # Reduced timeout
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 4:
                        gpu_name = parts[0].strip()
                        total_mb = float(parts[1].strip())
                        used_mb = float(parts[2].strip())
                        free_mb = float(parts[3].strip())
                        utilization_percent = (used_mb / total_mb) * 100
                        
                        gpu_info.update({
                            'has_gpu': True,
                            'total_mb': total_mb,
                            'used_mb': used_mb,
                            'free_mb': free_mb,
                            'utilization_percent': utilization_percent,
                            'gpu_name': gpu_name,
                            'method': 'nvidia-smi'
                        })
                        return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try Intel GPU detection via Windows Management Instrumentation (WMI)
        try:
            # Query for Intel integrated graphics
            wmi_result = subprocess.run([
                'powershell', '-Command',
                "Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like '*Intel*' -or $_.Name -like '*UHD*' -or $_.Name -like '*Iris*' -or $_.Name -like '*HD Graphics*'} | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ], capture_output=True, text=True, timeout=3)  # Reduced timeout
            
            if wmi_result.returncode == 0 and wmi_result.stdout.strip():
                wmi_data = json.loads(wmi_result.stdout.strip())
                
                # Handle both single GPU and multiple GPUs
                if isinstance(wmi_data, dict):
                    wmi_data = [wmi_data]
                
                for gpu_data in wmi_data:
                    if gpu_data.get('Name') and gpu_data.get('AdapterRAM'):
                        gpu_name = gpu_data['Name']
                        # AdapterRAM is in bytes, convert to MB
                        total_mb = int(gpu_data['AdapterRAM']) / (1024 * 1024)
                        
                        gpu_info.update({
                            'has_gpu': True,
                            'total_mb': total_mb,
                            'used_mb': 0,  # WMI doesn't provide usage info
                            'free_mb': total_mb,  # Assume all free since we can't get usage
                            'utilization_percent': 0,
                            'gpu_name': gpu_name,
                            'method': 'WMI_Intel'
                        })
                        logger.debug(f"Intel GPU detected via WMI: {gpu_name}, {total_mb:.0f}MB")
                        return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Try AMD GPU detection via WMI
        try:
            wmi_result = subprocess.run([
                'powershell', '-Command',
                "Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' -or $_.Name -like '*ATI*'} | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ], capture_output=True, text=True, timeout=3)  # Reduced timeout
            
            if wmi_result.returncode == 0 and wmi_result.stdout.strip():
                wmi_data = json.loads(wmi_result.stdout.strip())
                
                # Handle both single GPU and multiple GPUs
                if isinstance(wmi_data, dict):
                    wmi_data = [wmi_data]
                
                for gpu_data in wmi_data:
                    if gpu_data.get('Name') and gpu_data.get('AdapterRAM'):
                        gpu_name = gpu_data['Name']
                        # AdapterRAM is in bytes, convert to MB
                        total_mb = int(gpu_data['AdapterRAM']) / (1024 * 1024)
                        
                        gpu_info.update({
                            'has_gpu': True,
                            'total_mb': total_mb,
                            'used_mb': 0,  # WMI doesn't provide usage info
                            'free_mb': total_mb,  # Assume all free since we can't get usage
                            'utilization_percent': 0,
                            'gpu_name': gpu_name,
                            'method': 'WMI_AMD'
                        })
                        return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Generic WMI query for any video controller
        try:
            wmi_result = subprocess.run([
                'powershell', '-Command',
                "Get-WmiObject -Class Win32_VideoController | Where-Object {$_.AdapterRAM -gt 0} | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ], capture_output=True, text=True, timeout=3)  # Reduced timeout
            
            if wmi_result.returncode == 0 and wmi_result.stdout.strip():
                wmi_data = json.loads(wmi_result.stdout.strip())
                
                # Handle both single GPU and multiple GPUs
                if isinstance(wmi_data, dict):
                    wmi_data = [wmi_data]
                
                # Find the first GPU with significant memory (>= 512MB)
                for gpu_data in wmi_data:
                    if gpu_data.get('Name') and gpu_data.get('AdapterRAM'):
                        total_mb = int(gpu_data['AdapterRAM']) / (1024 * 1024)
                        
                        # Only consider GPUs with at least 512MB
                        if total_mb >= 512:
                            gpu_name = gpu_data['Name']
                            
                            gpu_info.update({
                                'has_gpu': True,
                                'total_mb': total_mb,
                                'used_mb': 0,  # WMI doesn't provide usage info
                                'free_mb': total_mb,  # Assume all free since we can't get usage
                                'utilization_percent': 0,
                                'gpu_name': gpu_name,
                                'method': 'WMI_Generic'
                            })
                            return gpu_info
                        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
            
    except Exception as e:
        logger.debug(f"Windows GPU detection methods failed: {e}")
    
    # Try DirectX DXGI detection (Windows 10+)
    try:
        # Use PowerShell to query DirectX information
        dxdiag_result = subprocess.run([
            'powershell', '-Command',
            """
            Add-Type -AssemblyName System.Windows.Forms
            $dxdiag = Get-WmiObject -Class Win32_VideoController | Where-Object {$_.AdapterRAM -gt 536870912} | Select-Object Name, AdapterRAM, DriverVersion
            $dxdiag | ConvertTo-Json
            """
        ], capture_output=True, text=True, timeout=3)  # Reduced timeout
        
        if dxdiag_result.returncode == 0 and dxdiag_result.stdout.strip():
            dx_data = json.loads(dxdiag_result.stdout.strip())
            
            if isinstance(dx_data, dict):
                dx_data = [dx_data]
            
            for gpu_data in dx_data:
                if gpu_data.get('Name') and gpu_data.get('AdapterRAM'):
                    gpu_name = gpu_data['Name']
                    total_mb = int(gpu_data['AdapterRAM']) / (1024 * 1024)
                    
                    gpu_info.update({
                        'has_gpu': True,
                        'total_mb': total_mb,
                        'used_mb': 0,
                        'free_mb': total_mb,
                        'utilization_percent': 0,
                        'gpu_name': gpu_name,
                        'method': 'DirectX_DXGI'
                    })
                    return gpu_info
                    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Try Linux/macOS methods for integrated GPUs
    try:
        system = platform.system().lower()
        
        if system == 'linux':
            # Try lspci for GPU detection on Linux
            try:
                lspci_result = subprocess.run(['lspci', '-v'], capture_output=True, text=True, timeout=2)  # Reduced timeout
                if lspci_result.returncode == 0:
                    output = lspci_result.stdout.lower()
                    
                    # Look for VGA/Display controllers
                    if any(keyword in output for keyword in ['vga', 'display', 'gpu', 'graphics']):
                        # Extract GPU names
                        lines = lspci_result.stdout.split('\n')
                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['vga', 'display controller', '3d controller']):
                                # Try to extract GPU name from the line
                                if 'intel' in line.lower():
                                    gpu_name = 'Intel Integrated Graphics (detected via lspci)'
                                elif 'amd' in line.lower() or 'ati' in line.lower():
                                    gpu_name = 'AMD Integrated Graphics (detected via lspci)'
                                elif 'nvidia' in line.lower():
                                    gpu_name = 'NVIDIA GPU (detected via lspci)'
                                else:
                                    gpu_name = 'GPU detected via lspci'
                                
                                gpu_info.update({
                                    'has_gpu': True,
                                    'total_mb': 0,  # Can't get memory info from lspci
                                    'used_mb': 0,
                                    'free_mb': 0,
                                    'utilization_percent': 0,
                                    'gpu_name': gpu_name,
                                    'method': 'lspci_linux'
                                })
                                return gpu_info
                                
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Try checking /sys/class/drm for Intel iGPU on Linux
            try:
                drm_path = '/sys/class/drm'
                if os.path.exists(drm_path):
                    drm_devices = os.listdir(drm_path)
                    intel_devices = [d for d in drm_devices if 'i915' in d or 'intel' in d.lower()]
                    
                    if intel_devices:
                        gpu_info.update({
                            'has_gpu': True,
                            'total_mb': 0,
                            'used_mb': 0,
                            'free_mb': 0,
                            'utilization_percent': 0,
                            'gpu_name': 'Intel Integrated Graphics (detected via /sys/class/drm)',
                            'method': 'linux_drm'
                        })
                        return gpu_info
                        
            except Exception:
                pass
        
        elif system == 'darwin':  # macOS
            # Try system_profiler for macOS GPU detection
            try:
                profiler_result = subprocess.run([
                    'system_profiler', 'SPDisplaysDataType', '-json'
                ], capture_output=True, text=True, timeout=15)
                
                if profiler_result.returncode == 0:
                    display_data = json.loads(profiler_result.stdout)
                    
                    displays = display_data.get('SPDisplaysDataType', [])
                    for display in displays:
                        gpu_name = display.get('sppci_model', 'Unknown GPU')
                        vram = display.get('spdisplays_vram', '0 MB')
                        
                        # Extract VRAM size
                        vram_mb = 0
                        if 'MB' in vram:
                            try:
                                vram_mb = float(vram.replace(' MB', ''))
                            except:
                                pass
                        elif 'GB' in vram:
                            try:
                                vram_mb = float(vram.replace(' GB', '')) * 1024
                            except:
                                pass
                        
                        gpu_info.update({
                            'has_gpu': True,
                            'total_mb': vram_mb,
                            'used_mb': 0,
                            'free_mb': vram_mb,
                            'utilization_percent': 0,
                            'gpu_name': gpu_name,
                            'method': 'macos_system_profiler'
                        })
                        return gpu_info
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                pass
                
    except Exception as e:
        logger.debug(f"Linux/macOS GPU detection failed: {e}")
    
    # Cache the result to prevent repeated expensive calls
    get_gpu_memory_info._cached_info = gpu_info
    get_gpu_memory_info._cache_time = current_time
    
    return gpu_info

CONFIG_DIR = 'scenarios'
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

class TemplateCache:
    """
    Cache for template images to avoid reloading them repeatedly.
    Uses weak references to allow garbage collection when templates are no longer needed.
    """
    def __init__(self, max_size=50):
        self._cache = {}
        self._max_size = max_size
        self._access_times = {}
    
    def get_template(self, path, force_reload=False):
        """Get template image from cache or load it."""
        if force_reload or path not in self._cache:
            if len(self._cache) >= self._max_size:
                self._cleanup_old_entries()
            
            template = cv2.imread(path)
            if template is not None:
                self._cache[path] = template
                self._access_times[path] = time.time()
            return template
        else:
            self._access_times[path] = time.time()
            return self._cache[path]
    
    def _cleanup_old_entries(self):
        """Remove least recently used entries."""
        if not self._access_times:
            return
        
        # Remove oldest 25% of entries
        sorted_entries = sorted(self._access_times.items(), key=lambda x: x[1])
        entries_to_remove = len(sorted_entries) // 4
        
        for path, _ in sorted_entries[:entries_to_remove]:
            self._cache.pop(path, None)
            self._access_times.pop(path, None)
    
    def clear(self):
        """Clear all cached templates."""
        self._cache.clear()
        self._access_times.clear()
        gc.collect()

# Global template cache instance
template_cache = TemplateCache()

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
    Optimized for better performance and memory usage.
    Returns a dict with x, y, width, height, or None.
    """
    root = tk.Tk()
    root.attributes("-alpha", 0.3)
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)  # Ensure window stays on top
    root.wait_visibility(root)
    
    # Configure for better performance
    root.resizable(False, False)
    root.overrideredirect(True)
    
    canvas = tk.Canvas(root, cursor="cross", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    rect = None
    start_x = None
    start_y = None
    selection_rect = None
    
    # Add instructions
    instruction_text = canvas.create_text(
        root.winfo_screenwidth() // 2, 50, 
        text="Click and drag to select area. Press ESC to cancel.", 
        fill="white", font=("Arial", 14)
    )

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x = event.x
        start_y = event.y
        # Remove instruction text
        canvas.delete(instruction_text)
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

    def on_mouse_drag(event):
        nonlocal rect
        if rect and start_x is not None and start_y is not None:
            cur_x, cur_y = event.x, event.y
            canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def on_button_release(event):
        nonlocal selection_rect
        if start_x is not None and start_y is not None:
            end_x, end_y = event.x, event.y
            
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            width = x2 - x1
            height = y2 - y1

            # Minimum size validation
            if width > 10 and height > 10:
                selection_rect = {"x": x1, "y": y1, "width": width, "height": height}
            else:
                selection_rect = None
        
        root.quit()

    def on_escape(event):
        nonlocal selection_rect
        selection_rect = None
        root.quit()
    
    def on_key_press(event):
        if event.keysym == 'Escape':
            on_escape(event)

    # Bind events
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    root.bind("<Escape>", on_escape)
    root.bind("<KeyPress>", on_key_press)
    
    # Focus the window to receive key events
    root.focus_set()
    root.focus_force()

    try:
        root.mainloop()
    except Exception as e:
        logger.error(f"Error in screenshot selection: {e}")
        selection_rect = None
    finally:
        try:
            root.destroy()
        except Exception as e:
            logger.warning(f"Error destroying tkinter window: {e}")
    
    return selection_rect


class SettingsDialog(QtWidgets.QDialog):
    """
    Settings dialog for application configuration.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle('Settings')
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        # Apply the same theme as parent window
        if parent:
            colors = parent.get_theme_styles()
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {colors['background']};
                    color: {colors['foreground']};
                }}
                QGroupBox {{
                    font-size: 9pt;
                    font-weight: bold;
                    border: 1px solid {colors['border']};
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 5px;
                    color: {colors['foreground']};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {colors['foreground']};
                }}
                QLabel {{
                    font-size: 9pt;
                    color: {colors['foreground']};
                }}
                QComboBox {{
                    font-size: 9pt;
                    min-height: 20px;
                    padding: 2px 6px;
                    min-width: 90px;
                    background-color: {colors['input_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    subcontrol-origin: padding;
                    subcontrol-position: center right;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    width: 0px;
                    height: 0px;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid {colors['foreground']};
                    margin: 0px;
                    padding: 0px;
                }}
                QComboBox::down-arrow:hover {{
                    border-top-color: {colors['accent']};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {colors['input_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                    selection-background-color: {colors['accent']};
                    selection-color: white;
                }}
                QPushButton {{
                    font-size: 9pt;
                    min-height: 20px;
                    padding: 2px 4px;
                    min-width: 70px;
                    background-color: {colors['button_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                }}
                QPushButton:hover {{
                    background-color: {colors['button_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {colors['border']};
                }}
            """)
            
            # Force immediate visual update for initial theme
            self.update()
            self.repaint()
        
        self.init_ui()
        
        # Apply theme to all child widgets after UI initialization
        if parent:
            QtCore.QTimer.singleShot(0, self.apply_theme_to_all_widgets)
    
    def init_ui(self):
        """Initialize the settings dialog UI."""
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Appearance settings group
        appearance_group = QtWidgets.QGroupBox('Appearance')
        appearance_layout = QtWidgets.QFormLayout()
        appearance_layout.setSpacing(8)
        
        # Theme setting
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.setMinimumWidth(120)
        self.theme_combo.setToolTip('Select theme: System (follows OS), Light, or Dark')
        self.theme_combo.addItem('System', 'system')
        self.theme_combo.addItem('Light', 'light')
        self.theme_combo.addItem('Dark', 'dark')
        
        # Set current theme
        if self.parent_window:
            theme_display_map = {'system': 'System', 'light': 'Light', 'dark': 'Dark'}
            current_theme_display = theme_display_map.get(self.parent_window.theme_mode, 'System')
            self.theme_combo.setCurrentText(current_theme_display)
        
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        appearance_layout.addRow('Theme:', self.theme_combo)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Logging settings group
        logging_group = QtWidgets.QGroupBox('Logging')
        logging_layout = QtWidgets.QFormLayout()
        logging_layout.setSpacing(8)
        
        # Logging level setting
        self.logging_combo = QtWidgets.QComboBox()
        self.logging_combo.setMinimumWidth(120)
        self.logging_combo.setToolTip('Set logging level: DEBUG shows all messages, ERROR shows only critical errors')
        
        logging_levels = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
            ('CRITICAL', logging.CRITICAL)
        ]
        
        for text, level in logging_levels:
            self.logging_combo.addItem(text, level)
        
        # Set current logging level
        if self.parent_window and hasattr(self.parent_window, 'logging_combo'):
            current_index = self.parent_window.logging_combo.currentIndex()
            self.logging_combo.setCurrentIndex(current_index)
        else:
            self.logging_combo.setCurrentIndex(1)  # Default to INFO
        
        self.logging_combo.currentTextChanged.connect(self.on_logging_level_changed)
        logging_layout.addRow('Log Level:', self.logging_combo)
        
        logging_group.setLayout(logging_layout)
        layout.addWidget(logging_group)
        
        # Resource monitoring settings group
        monitoring_group = QtWidgets.QGroupBox('Resource Monitoring')
        monitoring_layout = QtWidgets.QFormLayout()
        monitoring_layout.setSpacing(8)
        
        # Update interval setting
        self.update_interval_combo = QtWidgets.QComboBox()
        self.update_interval_combo.setMinimumWidth(120)
        self.update_interval_combo.setToolTip('Set resource monitoring update interval')
        
        intervals = [
            ('0.5s', 500),
            ('1s', 1000),
            ('2s', 2000),
            ('3s', 3000),
            ('5s', 5000),
            ('10s', 10000)
        ]
        
        for text, value in intervals:
            self.update_interval_combo.addItem(text, value)
        
        # Set current update interval
        if self.parent_window and hasattr(self.parent_window, 'update_interval_combo'):
            current_index = self.parent_window.update_interval_combo.currentIndex()
            self.update_interval_combo.setCurrentIndex(current_index)
        else:
            self.update_interval_combo.setCurrentIndex(2)  # Default to 2s
        
        self.update_interval_combo.currentIndexChanged.connect(self.on_update_interval_changed)
        monitoring_layout.addRow('Update Interval:', self.update_interval_combo)
        
        monitoring_group.setLayout(monitoring_layout)
        layout.addWidget(monitoring_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        # Close button
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def apply_theme_to_all_widgets(self):
        """Apply theme to all child widgets to ensure consistency."""
        if not self.parent_window:
            return
            
        # Force update all child widgets
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.update()
        
        # Force layout updates
        if self.layout():
            self.layout().update()
        
        # Process all events to ensure immediate visual update
        QtWidgets.QApplication.processEvents()
    
    def on_theme_changed(self, theme_text):
        """Handle theme change from settings dialog."""
        if self.parent_window:
            self.parent_window.on_theme_changed(theme_text)
            # Update the dialog's theme immediately with new colors
            colors = self.parent_window.get_theme_styles()
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {colors['background']};
                    color: {colors['foreground']};
                }}
                QGroupBox {{
                    font-size: 9pt;
                    font-weight: bold;
                    border: 1px solid {colors['border']};
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 5px;
                    color: {colors['foreground']};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {colors['foreground']};
                }}
                QLabel {{
                    font-size: 9pt;
                    color: {colors['foreground']};
                }}
                QComboBox {{
                    font-size: 9pt;
                    min-height: 20px;
                    padding: 2px 6px;
                    min-width: 90px;
                    background-color: {colors['input_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    subcontrol-origin: padding;
                    subcontrol-position: center right;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    width: 0px;
                    height: 0px;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid {colors['foreground']};
                    margin: 0px;
                    padding: 0px;
                }}
                QComboBox::down-arrow:hover {{
                    border-top-color: {colors['accent']};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {colors['input_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                    selection-background-color: {colors['accent']};
                    selection-color: white;
                }}
                QPushButton {{
                    font-size: 9pt;
                    min-height: 20px;
                    padding: 2px 4px;
                    min-width: 70px;
                    background-color: {colors['button_bg']};
                    border: 1px solid {colors['border']};
                    border-radius: 3px;
                    color: {colors['foreground']};
                }}
                QPushButton:hover {{
                    background-color: {colors['button_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {colors['border']};
                }}
            """)
            
            # Force immediate visual update
            self.update()
            self.repaint()
            
            # Process all pending events to ensure immediate application
            QtWidgets.QApplication.processEvents()
            
            # Apply theme to all child widgets
            self.apply_theme_to_all_widgets()
    
    def on_logging_level_changed(self, level_text):
        """Handle logging level change from settings dialog."""
        if self.parent_window:
            self.parent_window.on_logging_level_changed(level_text)
    
    def on_update_interval_changed(self):
        """Handle update interval change from settings dialog."""
        if self.parent_window:
            # Sync the parent's combo box
            parent_combo = self.parent_window.update_interval_combo
            parent_combo.setCurrentIndex(self.update_interval_combo.currentIndex())
            self.parent_window._on_update_interval_changed()


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for Scenario Image Automation.
    Handles UI, scenario management, and automation logic.
    """
    def automation_loop(self):
        """
        Optimized automation loop with memory management and performance improvements.
        """
        logger.info('Automation loop started.')
        frame_count = 0
        last_gc_time = time.time()
        
        try:
            while self.running:
                loop_start_time = time.time()
                
                # Only update the state label periodically to reduce UI overhead
                if frame_count % 10 == 0:
                    self.set_state('Looking for matches')
                
                # Get optimized screenshot
                screen_data = self._get_optimized_screenshot()
                if screen_data is None:
                    time.sleep(0.1)
                    continue
                
                screen_np, offset_x, offset_y = screen_data
                
                # Process steps with priority and cooldown system
                step_executed = False
                current_time = time.time()
                
                for step_idx, step in enumerate(self.current_scenario.steps):
                    # Check step cooldown to prevent spam
                    step_name = step.get('name', f'step_{step_idx}')
                    if step_name in self._step_cooldown:
                        if current_time - self._step_cooldown[step_name] < 1.0:
                            continue
                    
                    # Process step images with cache
                    detections = self._process_step_images(step, screen_np, offset_x, offset_y)
                    
                    # Check trigger condition
                    if self._check_step_trigger(step, detections):
                        self._processing_step = True
                        self.set_state(f'Performing step: {step_name}')
                        logger.info(f"Executing step: {step_name}")
                        
                        # Execute step actions
                        success = self._execute_step_actions(step, detections)
                        
                        if success:
                            self._step_cooldown[step_name] = current_time
                            step_executed = True
                            
                        self._processing_step = False
                        break  # Execute only one step per loop iteration
                
                # Performance monitoring
                loop_time = time.time() - loop_start_time
                self._update_performance_stats(loop_time)
                
                # Periodic cleanup
                frame_count += 1
                if frame_count % 100 == 0:  # Every 100 frames
                    self._periodic_cleanup()
                
                # Dynamic sleep based on performance
                sleep_time = max(0.05, 0.2 - loop_time) if not step_executed else 0.1
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f'Automation loop error: {e}')
        finally:
            self._processing_step = False
            self.set_state('Paused')
            logger.info('Automation loop ended.')
    
    def _get_optimized_screenshot(self):
        """
        Get screenshot with caching to reduce memory allocation.
        """
        current_time = time.time()
        
        # Use cached screenshot if recent enough and not processing
        if (self._last_screenshot is not None and 
            not self._processing_step and
            current_time - self._last_screenshot_time < self._screenshot_cache_duration):
            return self._last_screenshot
        
        try:
            selected_window = self.window_combo.currentText()
            if selected_window == 'Entire Screen':
                screen = pyautogui.screenshot()
                offset_x, offset_y = 0, 0
            else:
                # Try to get specific window
                win = None
                for w in gw.getAllWindows():
                    if w.title == selected_window:
                        win = w
                        break
                
                if win is not None and win.width > 0 and win.height > 0:
                    x, y, w_, h_ = win.left, win.top, win.width, win.height
                    # Validate window bounds
                    if x >= 0 and y >= 0 and w_ > 0 and h_ > 0:
                        screen = pyautogui.screenshot(region=(x, y, w_, h_))
                        offset_x, offset_y = x, y
                    else:
                        screen = pyautogui.screenshot()
                        offset_x, offset_y = 0, 0
                else:
                    screen = pyautogui.screenshot()
                    offset_x, offset_y = 0, 0
            
            # Convert to OpenCV format with optimized memory usage
            screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            
            # Cache the result
            screen_data = (screen_np, offset_x, offset_y)
            self._last_screenshot = screen_data
            self._last_screenshot_time = current_time
            
            # Clean up PIL image to free memory
            del screen
            
            return screen_data
            
        except Exception as e:
            logger.error(f'Error capturing screenshot: {e}')
            return None
    
    def _process_step_images(self, step, screen_np, offset_x, offset_y):
        """
        Process step images with template caching.
        """
        detections = {}
        
        for img in step.get('images', []):
            try:
                # Use cached template
                template = template_cache.get_template(img['path'])
                if template is None:
                    logger.warning(f"Could not load template image: {img['path']}")
                    continue
                
                # Perform template matching with optimization
                res = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                sensitivity = img.get('sensitivity', 0.9)
                if max_val > sensitivity:
                    # Adjust detected location to be relative to the full screen
                    abs_loc = (max_loc[0] + offset_x, max_loc[1] + offset_y)
                    detections[img['name']] = (abs_loc, template.shape, max_val)
                
                # Clean up OpenCV result to free memory
                del res
                
            except Exception as e:
                logger.error(f"Error in image detection for {img.get('name', 'unknown')}: {e}")
        
        return detections
    
    def _check_step_trigger(self, step, detections):
        """
        Check if step should be triggered based on detections.
        """
        step_images = step.get('images', [])
        if not step_images:
            return False
        
        found = [img for img in step_images if img.get('name') in detections]
        condition = step.get('condition', 'OR')
        
        if condition == 'AND':
            return len(found) == len(step_images)
        else:  # OR condition
            return len(found) > 0
    
    def _execute_step_actions(self, step, detections):
        """
        Execute step actions with error handling.
        """
        try:
            found_images = [img for img in step.get('images', []) if img.get('name') in detections]
            if found_images:
                # Use the first detected image for position reference
                ref_img = found_images[0]
                loc, shape, confidence = detections.get(ref_img.get('name'), ((0, 0), (0, 0, 0), 0))
                
                for action in step.get('actions', []):
                    self._perform_step_action(action, loc, shape)
                
                return True
            
        except Exception as e:
            logger.error(f"Error executing step actions: {e}")
        
        return False
    
    def _update_performance_stats(self, loop_time):
        """
        Update performance statistics for monitoring.
        """
        self._loop_times.append(loop_time)
        if len(self._loop_times) > self._max_loop_time_samples:
            self._loop_times.pop(0)
        
        # Performance warnings are now handled by _monitor_performance with deduplication
    
    def _periodic_cleanup(self):
        """
        Perform periodic cleanup to prevent memory leaks.
        """
        # Clear old screenshot cache
        self._last_screenshot = None
        
        # Clean up old step cooldowns (older than 5 minutes)
        current_time = time.time()
        old_cooldowns = [name for name, timestamp in self._step_cooldown.items() 
                        if current_time - timestamp > 300]
        for name in old_cooldowns:
            del self._step_cooldown[name]
        
        # Force garbage collection
        gc.collect()

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
        Start the automation process and worker thread with resource checks.
        """
        logger.info('Starting automation.')
        if not self.current_scenario or self.running:
            logger.warning('Start Automation: No scenario selected or already running.')
            return
        
        # Perform pre-start cleanup
        self._last_screenshot = None
        self._step_cooldown.clear()
        gc.collect()
        
        # Initialize performance monitoring
        self._loop_times.clear()
        
        # Start automation
        self.running = True
        self.set_state('Looking for matches')
        self.btn_start_stop.setText(f'Stop ({self.hotkey.upper()})')
        
        # Create and start worker thread
        self.worker = threading.Thread(target=self.automation_loop, daemon=True)
        self.worker.start()
        
        # Setup hotkey listener with error handling
        try:
            self.listener = keyboard.GlobalHotKeys({self.hotkey: self.stop_automation})
            self.listener.start()
        except Exception as e:
            logger.error(f"Failed to setup hotkey listener: {e}")
            # Continue without hotkey if setup fails
        
        logger.info('Automation started successfully.')
    
    def get_memory_usage(self):
        """
        Get current memory usage information for monitoring, including GPU memory.
        Uses timeout protection to prevent UI hanging.
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Get system info
            system_memory = psutil.virtual_memory()
            
            # Get GPU information with error handling (non-blocking)
            try:
                gpu_info = get_gpu_memory_info()
            except Exception as e:
                logger.debug(f"GPU detection failed: {e}")
                gpu_info = {'has_gpu': False, 'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization_percent': 0, 'gpu_name': 'Error', 'method': 'error'}
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,  # MB
                'process_memory_percent': process.memory_percent(),
                'cpu_percent': cpu_percent,
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024,  # GB
                'template_cache_size': len(template_cache._cache) if hasattr(template_cache, '_cache') else 0,
                'cooldown_entries': len(self._step_cooldown) if hasattr(self, '_step_cooldown') else 0,
                'has_psutil': True,
                # GPU information
                'gpu_has_gpu': gpu_info['has_gpu'],
                'gpu_total_mb': gpu_info['total_mb'],
                'gpu_used_mb': gpu_info['used_mb'],
                'gpu_free_mb': gpu_info['free_mb'],
                'gpu_utilization_percent': gpu_info['utilization_percent'],
                'gpu_name': gpu_info['gpu_name'],
                'gpu_method': gpu_info.get('method', 'none')
            }
        except ImportError:
            # psutil not available, minimal info without GPU detection to prevent hanging
            return {
                'process_memory_mb': 0,
                'process_memory_percent': 0,
                'cpu_percent': 0,
                'system_memory_percent': 0,
                'system_memory_available_gb': 0,
                'template_cache_size': len(template_cache._cache) if hasattr(template_cache, '_cache') else 0,
                'cooldown_entries': len(self._step_cooldown) if hasattr(self, '_step_cooldown') else 0,
                'has_psutil': False,
                # Minimal GPU info to prevent hanging
                'gpu_has_gpu': False,
                'gpu_total_mb': 0,
                'gpu_used_mb': 0,
                'gpu_free_mb': 0,
                'gpu_utilization_percent': 0,
                'gpu_name': 'psutil not available',
                'gpu_method': 'disabled'
            }
        except Exception as e:
            logger.warning(f"Error getting memory usage (non-critical): {e}")
            # Fallback minimal info to prevent hanging
            return {
                'process_memory_mb': 0,
                'process_memory_percent': 0,
                'cpu_percent': 0,
                'system_memory_percent': 0,
                'system_memory_available_gb': 0,
                'template_cache_size': len(template_cache._cache) if hasattr(template_cache, '_cache') else 0,
                'cooldown_entries': len(self._step_cooldown) if hasattr(self, '_step_cooldown') else 0,
                'has_psutil': False,
                'error': str(e),
                # Minimal GPU info to prevent hanging
                'gpu_has_gpu': False,
                'gpu_total_mb': 0,
                'gpu_used_mb': 0,
                'gpu_free_mb': 0,
                'gpu_utilization_percent': 0,
                'gpu_name': 'Error occurred',
                'gpu_method': 'error'
            }

    def stop_automation(self):
        """
        Stop the automation process and worker thread with proper cleanup.
        """
        logger.info('Stopping automation.')
        self.running = False
        self.last_toggle_time = time.time()
        self.set_state('Paused')
        self.btn_start_stop.setText(f'Start ({self.hotkey.upper()})')
        
        # Stop global hotkey listener
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                logger.warning(f"Error stopping hotkey listener: {e}")
            finally:
                self.listener = None
        
        # Wait for worker thread to finish (with timeout)
        if self.worker and self.worker.is_alive():
            try:
                self.worker.join(timeout=2.0)  # Wait up to 2 seconds
                if self.worker.is_alive():
                    logger.warning("Worker thread did not stop within timeout")
            except Exception as e:
                logger.warning(f"Error joining worker thread: {e}")
            finally:
                self.worker = None
        
        # Clear processing flags and cached data
        self._processing_step = False
        self._last_screenshot = None
        self._step_cooldown.clear()
        
        # Force garbage collection to free memory
        gc.collect()
        
        logger.info('Automation stopped and resources cleaned up.')

    def __init__(self):
        """
        Initialize the main window, UI, and load scenarios.
        """
        super().__init__()
        self.setWindowTitle('Scenario Image Automation')
        # Make window larger to accommodate all UI elements
        min_width = 640
        min_height = 720
        self.setMinimumSize(min_width, min_height)
        self.setGeometry(100, 100, min_width, min_height)
        self.running = False
        self.hotkey = '<f9>'
        self.listener = None
        self.worker = None
        self.current_scenario = None
        self.selected_step_idx = None
        self.last_toggle_time = 0  # For debounce of start/stop
        
        # Theme support
        self.theme_mode = 'system'  # 'light', 'dark', or 'system'
        
        # Memory optimization attributes
        self._last_screenshot = None
        self._last_screenshot_time = 0
        self._screenshot_cache_duration = 0.05  # Cache screenshot for 50ms
        self._processing_step = False
        self._step_cooldown = {}  # Cooldown tracking for steps
        
        # Performance monitoring
        self._loop_times = []
        self._max_loop_time_samples = 10
        self._last_warning_messages = {}  # Track last warning messages to prevent duplicates
        self._warning_cooldown = 30  # Seconds between duplicate warnings
        
        # Resource usage tracking for max/average values
        self._resource_history = {
            'memory_mb': [],
            'memory_percent': [],
            'cpu_percent': [],
            'system_memory_percent': [],
            'gpu_utilization': [],
            'loop_times': []
        }
        self._max_resource_samples = 50  # Keep last 50 samples for statistics
        
        self.init_ui()
        self.load_scenarios()
        
        # Restore window geometry (size and position)
        self._restore_window_geometry()
        
        # Apply theme after restoration (delayed to ensure window is ready)
        QtCore.QTimer.singleShot(100, self._apply_initial_theme)
        
        # Setup cleanup on close
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Setup periodic monitoring timer (disabled by default to prevent hanging)
        self.monitor_timer = QtCore.QTimer()
        self.monitor_timer.timeout.connect(self._monitor_performance)
        
        # Start with resource monitoring disabled to prevent startup hanging
        self.resource_monitoring_enabled = False
        
        # Initial resource display update (delayed to prevent startup hanging)
        QtCore.QTimer.singleShot(2000, self._enable_resource_monitoring)
        
        # Setup system theme monitoring timer (will be started by _apply_initial_theme if needed)
        self.theme_monitor_timer = QtCore.QTimer()
        self.theme_monitor_timer.timeout.connect(self._check_system_theme_change)
        self._last_system_theme = None
        
        # Save initial geometry after window is fully shown (delayed to ensure proper initialization)
        QtCore.QTimer.singleShot(1000, self._save_window_geometry)
    
    def _check_system_theme_change(self):
        """
        Check if the system theme has changed and update accordingly.
        Only runs when theme_mode is 'system'.
        """
        if self.theme_mode != 'system':
            self.theme_monitor_timer.stop()
            return
        
        try:
            current_system_theme = self.is_system_dark_theme()
            
            if self._last_system_theme is None:
                self._last_system_theme = current_system_theme
                return
            
            if current_system_theme != self._last_system_theme:
                logger.info(f'System theme changed to: {"Dark" if current_system_theme else "Light"}')
                self._last_system_theme = current_system_theme
                self.apply_theme()
                self._update_theme_combo_text()
                
        except Exception as e:
            logger.debug(f"Error checking system theme change: {e}")
    
    def _update_resource_history(self, memory_info):
        """
        Update resource usage history for calculating max/average values.
        """
        try:
            # Add current values to history
            if memory_info.get('has_psutil', False):
                self._resource_history['memory_mb'].append(memory_info['process_memory_mb'])
                self._resource_history['memory_percent'].append(memory_info['process_memory_percent'])
                self._resource_history['cpu_percent'].append(memory_info['cpu_percent'])
                self._resource_history['system_memory_percent'].append(memory_info['system_memory_percent'])
            
            if memory_info.get('gpu_has_gpu', False):
                self._resource_history['gpu_utilization'].append(memory_info['gpu_utilization_percent'])
            
            # Add loop time if available
            if hasattr(self, '_loop_times') and self._loop_times:
                avg_loop_time = sum(self._loop_times) / len(self._loop_times)
                self._resource_history['loop_times'].append(avg_loop_time * 1000)  # Convert to ms
            
            # Keep only the last N samples for each metric
            for key in self._resource_history:
                if len(self._resource_history[key]) > self._max_resource_samples:
                    self._resource_history[key] = self._resource_history[key][-self._max_resource_samples:]
                    
        except Exception as e:
            logger.debug(f"Error updating resource history: {e}")
    
    def _get_resource_stats(self, values_list):
        """
        Calculate current, max, and average values from a list of values.
        Returns (current, max, avg) or (0, 0, 0) if list is empty.
        """
        if not values_list:
            return 0, 0, 0
        
        current = values_list[-1] if values_list else 0
        max_val = max(values_list)
        avg_val = sum(values_list) / len(values_list)
        
        return current, max_val, avg_val
    
    def _apply_initial_theme(self):
        """
        Apply the initial theme and start theme monitoring after UI is ready.
        """
        # Apply the theme (includes title bar theming)
        self.apply_theme()
        self._update_theme_combo_text()
        
        # Apply title bar theming once more to ensure it takes effect
        is_dark = self.get_effective_dark_mode()
        self.apply_dark_title_bar(is_dark)
        
        # Start theme monitoring if in system mode
        if self.theme_mode == 'system':
            self._last_system_theme = self.is_system_dark_theme()
            self.theme_monitor_timer.start(5000)  # Check every 5 seconds
        
        logger.debug(f"Applied initial theme: {self.theme_mode}")
    
    def _enable_resource_monitoring(self):
        """Enable resource monitoring after startup delay."""
        if not hasattr(self, 'resource_monitoring_enabled') or not self.resource_monitoring_enabled:
            self.resource_monitoring_enabled = True
            
            # Get initial interval from dropdown (default is 2 seconds)
            initial_interval = self.update_interval_combo.currentData() or 2000
            self.monitor_timer.start(initial_interval)
            
            # Initial resource display update
            self._monitor_performance()
    
    def _monitor_performance(self):
        """
        Monitor application performance and memory usage.
        Uses error handling to prevent UI hanging.
        """
        # Check if monitoring is enabled
        if not getattr(self, 'resource_monitoring_enabled', False):
            return
            
        try:
            # Add timeout protection for memory info gathering
            memory_info = self.get_memory_usage()
            
            # Update resource display (this should be fast)
            self._update_resource_display(memory_info)
            
            # Log performance stats periodically (only if running)
            if self.running and hasattr(self, '_loop_times') and self._loop_times:
                avg_loop_time = sum(self._loop_times) / len(self._loop_times)
                
                # Warning thresholds with duplicate prevention
                if avg_loop_time > 1.0:
                    warning_msg = f"Performance warning: Average loop time {avg_loop_time:.3f}s"
                    self._log_warning_once("performance_slow", warning_msg)
                
                # Memory warning for cache-based monitoring
                if memory_info.get('template_cache_size', 0) > 20:
                    cache_msg = f"Template cache has {memory_info['template_cache_size']} entries"
                    self._log_warning_once("cache_large", cache_msg)
                
                # High memory usage warning
                if memory_info.get('process_memory_percent', 0) > 15:
                    memory_msg = f"High memory usage: {memory_info['process_memory_percent']:.1f}% of system memory"
                    self._log_warning_once("memory_high", memory_msg)
                
                # High CPU usage warning
                if memory_info.get('cpu_percent', 0) > 70:
                    cpu_msg = f"High CPU usage: {memory_info['cpu_percent']:.1f}%"
                    self._log_warning_once("cpu_high", cpu_msg)
                
                # GPU memory usage warning
                if memory_info.get('gpu_has_gpu', False) and memory_info.get('gpu_utilization_percent', 0) > 90:
                    gpu_msg = f"High GPU memory usage: {memory_info['gpu_utilization_percent']:.1f}%"
                    self._log_warning_once("gpu_high", gpu_msg)
                
        except Exception as e:
            logger.debug(f"Performance monitoring error: {e}")
            # If monitoring fails, show basic error info
            try:
                self.memory_label.setText("Memory: Monitor Error")
                self.cpu_label.setText("CPU: Monitor Error") 
                self.system_memory_label.setText("System: Monitor Error")
                self.cache_label.setText("Cache: Monitor Error")
                self.gpu_label.setText("GPU: Monitor Error")
                self.performance_label.setText(f"Monitor Error: {str(e)[:30]}...")
            except:
                pass  # If even setting error text fails, just ignore
    
    def _log_warning_once(self, warning_type, message):
        """
        Log a warning message only once per cooldown period to prevent spam.
        """
        current_time = time.time()
        
        # Check if we've already logged this type of warning recently
        if warning_type in self._last_warning_messages:
            last_time, last_message = self._last_warning_messages[warning_type]
            
            # If the message is the same and within cooldown period, skip
            if (current_time - last_time < self._warning_cooldown and 
                message == last_message):
                return
        
        # Log the warning and update tracking
        logger.warning(message)
        self._last_warning_messages[warning_type] = (current_time, message)
    
    def _update_resource_display(self, memory_info):
        """
        Update the resource usage display in the UI with theme-aware colors and max/average statistics.
        """
        try:
            colors = self.get_theme_styles()
            
            # Update resource history for statistics
            self._update_resource_history(memory_info)
            
            # Memory usage with statistics
            if memory_info.get('has_psutil', False):
                current_mb, max_mb, avg_mb = self._get_resource_stats(self._resource_history['memory_mb'])
                current_pct, max_pct, avg_pct = self._get_resource_stats(self._resource_history['memory_percent'])
                
                memory_text = f"Memory: {current_mb:.1f}MB ({current_pct:.1f}%)\nAvg: {avg_mb:.1f}MB ({avg_pct:.1f}%) | Max: {max_mb:.1f}MB ({max_pct:.1f}%)"
                memory_color = colors['error'] if current_pct > 10 else colors['success']
            else:
                memory_text = "Memory: N/A (psutil needed)\nInstall psutil for detailed monitoring"
                memory_color = colors['warning']
            
            self.memory_label.setText(memory_text)
            self.memory_label.setStyleSheet(f'font-size: 9pt; color: {memory_color};')
            
            # Make memory label clickable to show psutil installation info
            if not memory_info.get('has_psutil', False):
                self.memory_label.mousePressEvent = lambda event: self._show_psutil_info()
                self.memory_label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            
            # CPU usage with statistics
            if memory_info.get('has_psutil', False):
                current_cpu, max_cpu, avg_cpu = self._get_resource_stats(self._resource_history['cpu_percent'])
                
                cpu_text = f"CPU: {current_cpu:.1f}%\nAvg: {avg_cpu:.1f}% | Max: {max_cpu:.1f}%"
                cpu_color = colors['error'] if current_cpu > 50 else colors['success']
            else:
                cpu_text = "CPU: N/A\nInstall psutil for monitoring"
                cpu_color = colors['text_light']
            
            self.cpu_label.setText(cpu_text)
            self.cpu_label.setStyleSheet(f'font-size: 9pt; color: {cpu_color};')
            
            # System memory with statistics
            if memory_info.get('has_psutil', False):
                current_sys, max_sys, avg_sys = self._get_resource_stats(self._resource_history['system_memory_percent'])
                
                system_text = f"System: {current_sys:.1f}% ({memory_info['system_memory_available_gb']:.1f}GB free)\nAvg: {avg_sys:.1f}% | Max: {max_sys:.1f}%"
                system_color = colors['error'] if current_sys > 85 else colors['success']
            else:
                system_text = "System: N/A\nInstall psutil for monitoring"
                system_color = colors['text_light']
            
            self.system_memory_label.setText(system_text)
            self.system_memory_label.setStyleSheet(f'font-size: 9pt; color: {system_color};')
            
            # Cache info (enhanced display)
            cache_text = f"Cache: {memory_info['template_cache_size']} templates\n{memory_info['cooldown_entries']} cooldowns"
            cache_color = colors['warning'] if memory_info['template_cache_size'] > 20 else colors['info']
            
            self.cache_label.setText(cache_text)
            self.cache_label.setStyleSheet(f'font-size: 9pt; color: {cache_color};')
            
            # GPU memory info with statistics
            gpu_method = memory_info.get('gpu_method', 'unknown')
            
            if memory_info.get('gpu_has_gpu', False):
                gpu_used_mb = memory_info.get('gpu_used_mb', 0)
                gpu_total_mb = memory_info.get('gpu_total_mb', 0)
                gpu_utilization = memory_info.get('gpu_utilization_percent', 0)
                
                # Get GPU statistics
                current_gpu, max_gpu, avg_gpu = self._get_resource_stats(self._resource_history['gpu_utilization'])
                
                if gpu_total_mb > 0:
                    gpu_text = f"GPU: {gpu_used_mb:.0f}/{gpu_total_mb:.0f}MB ({current_gpu:.1f}%)\nAvg: {avg_gpu:.1f}% | Max: {max_gpu:.1f}%"
                    gpu_color = colors['error'] if current_gpu > 80 else colors['warning'] if current_gpu > 60 else colors['success']
                    
                    # Add GPU name as tooltip
                    gpu_name = memory_info.get('gpu_name', 'Unknown GPU')
                    self.gpu_label.setToolTip(f"GPU: {gpu_name} (detected via {gpu_method})")
                else:
                    gpu_text = f"GPU: Detected ({gpu_method})\nUtilization: {current_gpu:.1f}%"
                    gpu_color = colors['info']
                    gpu_name = memory_info.get('gpu_name', 'Unknown GPU')
                    self.gpu_label.setToolTip(f"GPU: {gpu_name} (limited info via {gpu_method})")
            elif gpu_method == 'loading':
                gpu_text = "GPU: Loading...\nDetection in progress"
                gpu_color = colors['warning']
                self.gpu_label.setToolTip("GPU detection in progress...")
            else:
                gpu_text = "GPU: Not detected\nClick for setup info"
                gpu_color = colors['text_light']
                self.gpu_label.setToolTip(
                    "No GPU detected or GPU monitoring libraries not available.\n\n"
                    "For enhanced GPU monitoring:\n"
                    "• NVIDIA GPUs: pip install pynvml or GPUtil\n"
                    "• Integrated GPUs: Built-in Windows WMI support\n"
                    "• Linux: lspci and DRM detection\n"
                    "• macOS: system_profiler integration\n\n"
                    "Click for more information about GPU monitoring."
                )
            
            self.gpu_label.setText(gpu_text)
            self.gpu_label.setStyleSheet(f'font-size: 9pt; color: {gpu_color};')
            
            # Make GPU label clickable to show GPU info when no GPU detected or loading
            if not memory_info.get('gpu_has_gpu', False) and gpu_method != 'loading':
                self.gpu_label.mousePressEvent = lambda event: self._show_gpu_info()
                self.gpu_label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            else:
                # Remove click handler if GPU is detected or loading
                self.gpu_label.mousePressEvent = None
                self.gpu_label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
            
            # Performance info with statistics
            if hasattr(self, '_loop_times') and self._loop_times:
                current_loop, max_loop, avg_loop = self._get_resource_stats(self._resource_history['loop_times'])
                
                perf_text = f"Performance: {current_loop:.0f}ms loop time\nAvg: {avg_loop:.0f}ms | Max: {max_loop:.0f}ms"
                perf_color = colors['error'] if current_loop > 500 else colors['success']
            else:
                perf_text = "Performance: Not running\nStart automation to see metrics"
                perf_color = colors['text_light']
            
            self.performance_label.setText(perf_text)
            self.performance_label.setStyleSheet(f'font-size: 9pt; color: {perf_color};')
            
            # Show error if any
            if 'error' in memory_info:
                self.performance_label.setText(f"Error: {memory_info['error'][:30]}...\nCheck logs for details")
                self.performance_label.setStyleSheet(f'font-size: 9pt; color: {colors["error"]};')
                
        except Exception as e:
            logger.debug(f"Error updating resource display: {e}")
            colors = self.get_theme_styles()
            # Show basic fallback info with theme-aware colors
            error_color = colors['error']
            self.memory_label.setText("Memory: Error\nCheck logs for details")
            self.memory_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
            self.cpu_label.setText("CPU: Error\nCheck logs for details")
            self.cpu_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
            self.system_memory_label.setText("System: Error\nCheck logs for details")
            self.system_memory_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
            self.cache_label.setText("Cache: Error\nCheck logs for details")
            self.cache_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
            self.gpu_label.setText("GPU: Error\nCheck logs for details")
            self.gpu_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
            self.performance_label.setText(f"Display Error: {str(e)[:20]}...\nResource monitoring failed")
            self.performance_label.setStyleSheet(f'font-size: 9pt; color: {error_color};')
    
    def stop_monitoring(self):
        """Stop performance monitoring and theme monitoring timers."""
        if hasattr(self, 'monitor_timer') and self.monitor_timer:
            self.monitor_timer.stop()
        if hasattr(self, 'theme_monitor_timer') and self.theme_monitor_timer:
            self.theme_monitor_timer.stop()
    
    def _show_psutil_info(self):
        """Show information about installing psutil for enhanced monitoring."""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Enhanced Resource Monitoring")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setText("Install psutil for detailed resource monitoring")
        msg.setInformativeText(
            "For detailed CPU, memory, and system resource monitoring, install the psutil package:\n\n"
            "pip install psutil\n\n"
            "This will enable:\n"
            "• Process memory usage in MB and percentage\n"
            "• CPU usage percentage\n"
            "• System memory statistics\n"
            "• Available system memory\n\n"
            "Without psutil, only basic cache information is shown."
        )
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def _show_gpu_info(self):
        """Show information about installing GPU monitoring libraries."""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("GPU Memory Monitoring")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setText("GPU monitoring supports both discrete and integrated GPUs")
        msg.setInformativeText(
            "GPU Memory Monitoring Support:\n\n"
            "NVIDIA GPUs (Discrete):\n"
            "• pip install pynvml (recommended)\n"
            "• pip install GPUtil (alternative)\n"
            "• nvidia-smi command line tool\n\n"
            "Integrated GPUs (Intel/AMD):\n"
            "• Windows: WMI queries (built-in)\n"
            "• Linux: lspci and /sys/class/drm\n"
            "• macOS: system_profiler\n\n"
            "Features enabled:\n"
            "• GPU detection and identification\n"
            "• VRAM/memory size (where available)\n"
            "• Real-time usage (NVIDIA with libraries)\n"
            "• Cross-platform support\n\n"
            "Note: Integrated GPUs may show limited information\n"
            "compared to discrete GPUs due to system limitations.\n"
            "NVIDIA GPUs with proper libraries provide the most\n"
            "detailed monitoring including real-time usage."
        )
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def _on_update_interval_changed(self):
        """Handle resource monitoring update interval change."""
        interval = self.update_interval_combo.currentData()
        if interval and hasattr(self, 'monitor_timer'):
            # Stop current timer
            self.monitor_timer.stop()
            
            # Start timer with new interval
            self.monitor_timer.start(interval)
            
            logger.debug(f"Resource monitoring interval changed to {interval}ms")
            
            # Immediately update display to show the change is active
            QtCore.QTimer.singleShot(50, self._monitor_performance)
    
    def _save_window_geometry(self):
        """Save the current window size and position to a config file."""
        try:
            # Don't save if window is minimized or not visible
            if self.isMinimized() or not self.isVisible():
                return
                
            config_path = os.path.join(CONFIG_DIR, 'window_config.json')
            
            # Get current geometry
            geometry_data = {
                'x': self.x(),
                'y': self.y(),
                'width': self.width(),
                'height': self.height(),
                'maximized': self.isMaximized()
            }
            
            # Validate geometry data before saving - only prevent extremely invalid values
            if (geometry_data['width'] < 300 or geometry_data['height'] < 200 or
                geometry_data['width'] > 10000 or geometry_data['height'] > 10000):
                logger.debug(f"Invalid geometry dimensions, skipping save: {geometry_data}")
                return
            
            # Load existing config if it exists
            existing_config = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        existing_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_config = {}
            
            # Update with window geometry and settings
            existing_config['main_window'] = geometry_data
            existing_config['update_interval'] = self.update_interval_combo.currentData()
            existing_config['theme_mode'] = self.theme_mode
            existing_config['logging_level'] = self.logging_combo.currentData()
            
            # Create directory if it doesn't exist
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            # Write config file
            with open(config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)
            
            logger.debug(f"Saved window geometry and settings: {geometry_data}")
            
        except Exception as e:
            logger.debug(f"Error saving window geometry: {e}")
    
    def _restore_window_geometry(self):
        """Restore the window size and position from config file."""
        try:
            config_path = os.path.join(CONFIG_DIR, 'window_config.json')
            
            if not os.path.exists(config_path):
                logger.debug("No window config file found, using default geometry")
                return
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            geometry_data = config.get('main_window')
            if not geometry_data:
                logger.debug("No main window geometry data found")
                return
            
            # Validate geometry data
            required_keys = ['x', 'y', 'width', 'height']
            if not all(key in geometry_data for key in required_keys):
                logger.debug("Incomplete geometry data, using defaults")
                return
            
            # Additional validation for reasonable values
            if (geometry_data['width'] < 400 or geometry_data['height'] < 300 or
                geometry_data['width'] > 5000 or geometry_data['height'] > 5000):
                logger.debug("Invalid geometry dimensions, using defaults")
                return
            
            # Get screen dimensions to validate position
            screen = QtWidgets.QApplication.primaryScreen().geometry()
            screen_width = screen.width()
            screen_height = screen.height()
            
            # Use the exact saved position and dimensions - no adjustments
            x = geometry_data['x']
            y = geometry_data['y'] 
            width = geometry_data['width']
            height = geometry_data['height']
            
            # Only ensure the saved dimensions meet actual minimum requirements
            # but preserve the exact position
            if width < 640:
                width = 640
            if height < 720:
                height = 720
            
            # Only adjust position if window would be completely invisible (very permissive)
            # Allow negative positions for multi-monitor setups
            if x + width < 10:  # Less than 10px visible horizontally
                x = 10 - width
            if y + height < 10:  # Less than 10px visible vertically  
                y = 10 - height
            if x > screen_width - 10:  # Too far right (only if less than 10px visible)
                x = screen_width - 10
            if y > screen_height - 10:  # Too far down (only if less than 10px visible)
                y = screen_height - 10
            
            # Apply geometry immediately
            self.setGeometry(x, y, width, height)
            
            # Log the exact values being applied for debugging
            logger.debug(f"Applying geometry: x={x}, y={y}, width={width}, height={height}")
            logger.debug(f"Original saved position: x={geometry_data['x']}, y={geometry_data['y']}")
            logger.debug(f"Screen dimensions: {screen_width}x{screen_height}")
            
            # Apply geometry again with a small delay to override any system adjustments
            QtCore.QTimer.singleShot(50, lambda: self.setGeometry(x, y, width, height))
            
            # Restore maximized state if applicable
            if geometry_data.get('maximized', False):
                self.showMaximized()
            else:
                # Ensure we're not maximized and apply the geometry one more time
                QtCore.QTimer.singleShot(100, lambda: self.setGeometry(x, y, width, height))
            
            # Restore update interval if saved
            saved_interval = config.get('update_interval')
            if saved_interval:
                # Find the index of the saved interval in the combo box
                for i in range(self.update_interval_combo.count()):
                    if self.update_interval_combo.itemData(i) == saved_interval:
                        self.update_interval_combo.setCurrentIndex(i)
                        break
                logger.debug(f"Restored update interval: {saved_interval}ms")
            
            # Restore theme mode if saved
            saved_theme = config.get('theme_mode')
            if saved_theme and saved_theme in ['system', 'light', 'dark']:
                self.theme_mode = saved_theme
                # Update the combo box to match the restored theme
                theme_display_map = {'system': 'System', 'light': 'Light', 'dark': 'Dark'}
                theme_display = theme_display_map.get(saved_theme, 'System')
                self.theme_combo.setCurrentText(theme_display)
                logger.debug(f"Restored theme mode: {saved_theme}")
            
            # Restore logging level if saved
            saved_logging_level = config.get('logging_level')
            if saved_logging_level is not None:
                # Find the index of the saved logging level in the combo box
                for i in range(self.logging_combo.count()):
                    if self.logging_combo.itemData(i) == saved_logging_level:
                        self.logging_combo.setCurrentIndex(i)
                        # Apply the logging level immediately
                        logger.setLevel(saved_logging_level)
                        root_logger = logging.getLogger()
                        root_logger.setLevel(saved_logging_level)
                        for handler in logger.handlers:
                            handler.setLevel(saved_logging_level)
                        for handler in root_logger.handlers:
                            handler.setLevel(saved_logging_level)
                        break
                logger.debug(f"Restored logging level: {saved_logging_level}")
            
            logger.debug(f"Restored window geometry: x={x}, y={y}, w={width}, h={height}")
            
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.debug(f"Error loading window geometry: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error restoring window geometry: {e}")
    
    def moveEvent(self, event):
        """Handle window move events to save position."""
        super().moveEvent(event)
        # Save geometry after a short delay to avoid excessive writes during dragging
        if hasattr(self, '_geometry_save_timer'):
            self._geometry_save_timer.stop()
        else:
            self._geometry_save_timer = QtCore.QTimer()
            self._geometry_save_timer.setSingleShot(True)
            self._geometry_save_timer.timeout.connect(self._save_window_geometry)
        
        self._geometry_save_timer.start(500)  # Save after 500ms of no movement
    
    def resizeEvent(self, event):
        """Handle window resize events to save size."""
        super().resizeEvent(event)
        # Save geometry after a short delay to avoid excessive writes during resizing
        if hasattr(self, '_geometry_save_timer'):
            self._geometry_save_timer.stop()
        else:
            self._geometry_save_timer = QtCore.QTimer()
            self._geometry_save_timer.setSingleShot(True)
            self._geometry_save_timer.timeout.connect(self._save_window_geometry)
        
        self._geometry_save_timer.start(500)  # Save after 500ms of no resizing
    
    def showEvent(self, event):
        """Handle window show event to ensure title bar theming is applied."""
        super().showEvent(event)
        # Apply title bar theming when window is shown (once only)
        is_dark = self.get_effective_dark_mode()
        QtCore.QTimer.singleShot(50, lambda: self.apply_dark_title_bar(is_dark))
    
    def changeEvent(self, event):
        """Handle window state changes (maximize/minimize)."""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            # Save geometry when window state changes
            QtCore.QTimer.singleShot(100, self._save_window_geometry)
    
    def closeEvent(self, event):
        """Handle application close event with proper cleanup."""
        # Save window geometry before closing
        self._save_window_geometry()
        
        self.cleanup_resources()
        event.accept()
    
    def cleanup_resources(self):
        """Clean up resources to prevent memory leaks."""
        if self.running:
            self.stop_automation()
        
        # Stop monitoring timer
        self.stop_monitoring()
        
        # Clear template cache
        template_cache.clear()
        
        # Clear screenshot cache
        self._last_screenshot = None
        
        # Clear performance tracking
        if hasattr(self, '_loop_times'):
            self._loop_times.clear()
        if hasattr(self, '_step_cooldown'):
            self._step_cooldown.clear()
        if hasattr(self, '_resource_history'):
            for key in self._resource_history:
                self._resource_history[key].clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Resources cleaned up successfully")

    def is_system_dark_theme(self):
        """
        Detect if the system is using dark theme.
        Works on Windows 10/11, macOS, and some Linux desktop environments.
        """
        try:
            import platform
            system = platform.system().lower()
            
            if system == 'windows':
                # Windows 10/11 dark mode detection
                try:
                    import winreg
                    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                    key = winreg.OpenKey(registry, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    winreg.CloseKey(key)
                    return value == 0  # 0 = dark theme, 1 = light theme
                except (ImportError, FileNotFoundError, OSError):
                    # Fallback: check if PyQt6 can detect system dark theme
                    try:
                        palette = QtWidgets.QApplication.palette()
                        # Check if the window background is darker than text color
                        bg_color = palette.color(QtGui.QPalette.ColorRole.Window)
                        text_color = palette.color(QtGui.QPalette.ColorRole.WindowText)
                        # Calculate luminance to determine if background is dark
                        bg_luminance = (0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()) / 255
                        return bg_luminance < 0.5
                    except:
                        pass
            
            elif system == 'darwin':  # macOS
                try:
                    import subprocess
                    result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], 
                                          capture_output=True, text=True, timeout=2)
                    return result.stdout.strip().lower() == 'dark'
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            elif system == 'linux':
                # Try to detect dark theme on Linux (GNOME, KDE, etc.)
                try:
                    import subprocess
                    import os
                    
                    # Try GNOME gsettings
                    try:
                        result = subprocess.run(['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'], 
                                              capture_output=True, text=True, timeout=2)
                        theme_name = result.stdout.strip().strip("'").lower()
                        return any(dark_indicator in theme_name for dark_indicator in ['dark', 'adwaita-dark', 'breeze-dark'])
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                    
                    # Try KDE configuration
                    kde_config_paths = [
                        os.path.expanduser('~/.config/kdeglobals'),
                        os.path.expanduser('~/.kde/share/config/kdeglobals'),
                        os.path.expanduser('~/.kde4/share/config/kdeglobals')
                    ]
                    
                    for config_path in kde_config_paths:
                        try:
                            if os.path.exists(config_path):
                                with open(config_path, 'r') as f:
                                    content = f.read().lower()
                                    if 'colorscheme=breezedark' in content or 'dark' in content:
                                        return True
                        except:
                            continue
                    
                    # Fallback: check environment variables
                    desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                    if 'dark' in desktop_env:
                        return True
                        
                except:
                    pass
        
        except Exception as e:
            logger.debug(f"System theme detection failed: {e}")
        
        # Default fallback: assume light theme
        return False
    
    def get_effective_dark_mode(self):
        """
        Get the effective dark mode state based on current theme setting.
        """
        if self.theme_mode == 'dark':
            return True
        elif self.theme_mode == 'light':
            return False
        else:  # system
            return self.is_system_dark_theme()

    def get_theme_styles(self):
        """
        Get the current theme styles (light or dark mode).
        """
        is_dark = self.get_effective_dark_mode()
        
        if is_dark:
            return {
                'background': '#2b2b2b',
                'foreground': '#ffffff',
                'button_bg': '#404040',
                'button_hover': '#505050',
                'input_bg': '#3c3c3c',
                'border': '#555555',
                'accent': '#0078d4',
                'success': '#107c10',
                'warning': '#ff8c00',
                'error': '#d13438',
                'info': '#0078d4',
                'text_light': '#cccccc',
                'text_dark': '#ffffff'
            }
        else:
            return {
                'background': '#ffffff',
                'foreground': '#000000',
                'button_bg': '#f0f0f0',
                'button_hover': '#e0e0e0',
                'input_bg': '#ffffff',
                'border': '#cccccc',
                'accent': '#0055aa',
                'success': '#107c10',
                'warning': '#ff8c00',
                'error': '#d9534f',
                'info': '#0055aa',
                'text_light': '#333333',
                'text_dark': '#000000'
            }
    
    def apply_dark_title_bar(self, enable_dark=True):
        """
        Apply dark mode to the window title bar.
        Works on Windows 10/11, macOS, and some Linux desktop environments.
        """
        try:
            import platform
            system = platform.system().lower()
            
            if system == 'windows':
                # Windows 10/11 dark title bar
                try:
                    import ctypes
                    from ctypes import wintypes
                    
                    # Get window handle
                    hwnd = int(self.winId())
                    
                    # Define constants for Windows API
                    DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
                    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                    
                    # Check Windows version to determine which attribute to use
                    import sys
                    windows_version = sys.getwindowsversion()
                    
                    # Windows 11 or Windows 10 version 2004 and later use the newer attribute
                    use_newer_api = (windows_version.major > 10 or 
                                   (windows_version.major == 10 and windows_version.build >= 19041))
                    
                    attribute = DWMWA_USE_IMMERSIVE_DARK_MODE if use_newer_api else DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1
                    
                    try:
                        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                            hwnd,
                            attribute,
                            ctypes.byref(ctypes.c_int(1 if enable_dark else 0)),
                            ctypes.sizeof(ctypes.c_int)
                        )
                        
                        if result == 0:  # S_OK
                            logger.debug(f"Applied Windows {'dark' if enable_dark else 'light'} title bar ({'newer' if use_newer_api else 'legacy'} API)")
                            
                            # Force immediate window redraw
                            self.update()
                            self.repaint()
                            
                            return True
                        else:
                            logger.debug(f"Windows title bar API returned error code: {result}")
                            
                    except Exception as e:
                        logger.debug(f"Windows {'dark' if enable_dark else 'light'} title bar API call failed: {e}")
                        
                        # Try the other API as fallback
                        fallback_attribute = DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 if use_newer_api else DWMWA_USE_IMMERSIVE_DARK_MODE
                        try:
                            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                                hwnd,
                                fallback_attribute,
                                ctypes.byref(ctypes.c_int(1 if enable_dark else 0)),
                                ctypes.sizeof(ctypes.c_int)
                            )
                            if result == 0:
                                logger.debug(f"Applied Windows {'dark' if enable_dark else 'light'} title bar (fallback API)")
                                self.update()
                                self.repaint()
                                
                                return True
                        except Exception as fallback_e:
                            logger.debug(f"Windows dark title bar fallback failed: {fallback_e}")
                            
                except ImportError:
                    logger.debug("ctypes not available for Windows title bar theming")
                except Exception as e:
                    logger.debug(f"Windows title bar theming error: {e}")
            
            elif system == 'darwin':  # macOS
                try:
                    # For macOS, we could use PyObjC bindings if available
                    # This is a simplified approach - full implementation would require PyObjC
                    logger.debug("macOS title bar theming requires PyObjC bindings (not implemented)")
                    
                    # Alternative: Set window flags that might influence appearance
                    # This is limited but might work in some cases
                    if enable_dark:
                        # Try to hint to the system that we prefer dark appearance
                        pass
                        
                except Exception as e:
                    logger.debug(f"macOS title bar theming failed: {e}")
            
            elif system == 'linux':
                # Linux title bar theming depends on the window manager/desktop environment
                try:
                    desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                    
                    # For X11 systems, we might be able to set window properties
                    if os.environ.get('DISPLAY'):
                        try:
                            # This would require X11 libraries and is highly dependent on WM
                            logger.debug(f"Linux X11 desktop environment: {desktop_env} - title bar theming limited")
                        except Exception as e:
                            logger.debug(f"Linux X11 title bar theming failed: {e}")
                    
                    # For Wayland, title bar theming is even more limited
                    elif os.environ.get('WAYLAND_DISPLAY'):
                        logger.debug("Linux Wayland - title bar theming not supported")
                        
                except Exception as e:
                    logger.debug(f"Linux title bar theming failed: {e}")
        
        except Exception as e:
            logger.debug(f"Title bar theming failed: {e}")
        
        return False
    
    def apply_theme(self):
        """
        Apply the current theme to all UI elements including title bar.
        """
        colors = self.get_theme_styles()
        is_dark = self.get_effective_dark_mode()
        
        # Apply title bar theming immediately
        self.apply_dark_title_bar(is_dark)
        
        # Also apply to any open dialogs immediately
        for dialog in QtWidgets.QApplication.allWidgets():
            if isinstance(dialog, QtWidgets.QDialog) and dialog.isVisible():
                try:
                    hwnd = int(dialog.winId())
                    if hwnd:
                        import ctypes
                        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                        DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
                        
                        # Try both API versions
                        for attribute in [DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1]:
                            try:
                                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                                    hwnd,
                                    attribute,
                                    ctypes.byref(ctypes.c_int(1 if is_dark else 0)),
                                    ctypes.sizeof(ctypes.c_int)
                                )
                                if result == 0:
                                    dialog.update()
                                    dialog.repaint()
                                    break
                            except:
                                continue
                except:
                    pass
        
        # Main window style
        main_style = f"""
            QMainWindow {{
                background-color: {colors['background']};
                color: {colors['foreground']};
            }}
            QPushButton {{
                font-size: 9pt;
                min-height: 20px;
                padding: 2px 4px;
                min-width: 35px;
                background-color: {colors['button_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                color: {colors['foreground']};
            }}
            QPushButton:hover {{
                background-color: {colors['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['border']};
            }}
            QComboBox {{
                font-size: 9pt;
                min-height: 20px;
                padding: 2px 6px;
                min-width: 90px;
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                color: {colors['foreground']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {colors['foreground']};
                margin: 0px;
                padding: 0px;
            }}
            QComboBox::down-arrow:hover {{
                border-top-color: {colors['accent']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                color: {colors['foreground']};
                selection-background-color: {colors['accent']};
                selection-color: white;
            }}
            QListWidget {{
                font-size: 9pt;
                min-width: 200px;
                min-height: 120px;
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                color: {colors['foreground']};
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {colors['border']};
            }}
            QListWidget::item:selected {{
                background-color: {colors['accent']};
                color: white;
            }}
            QLabel {{
                font-size: 9pt;
                color: {colors['foreground']};
            }}
            QGroupBox {{
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid {colors['border']};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
                color: {colors['foreground']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {colors['foreground']};
            }}
            QWidget {{
                color: {colors['foreground']};
                background-color: {colors['background']};
            }}
        """
        
        self.setStyleSheet(main_style)
        
        # Update individual labels with theme-appropriate colors
        self.state_label.setStyleSheet(f'font-weight: bold; font-size: 11pt; color: {colors["accent"]};')
        
        # Update resource labels
        self.memory_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
        self.cpu_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
        self.system_memory_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
        self.cache_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
        self.gpu_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
        self.performance_label.setStyleSheet(f'font-size: 9pt; color: {colors["text_light"]};')
    
    def on_theme_changed(self, theme_text):
        """
        Handle theme dropdown selection change.
        """
        # Map display text to internal theme mode
        theme_map = {
            'System': 'system',
            'Light': 'light', 
            'Dark': 'dark'
        }
        
        new_theme = theme_map.get(theme_text, 'system')
        
        if new_theme != self.theme_mode:
            self.theme_mode = new_theme
            
            # Start/stop theme monitoring based on mode
            if self.theme_mode == 'system':
                self._last_system_theme = self.is_system_dark_theme()
                if hasattr(self, 'theme_monitor_timer'):
                    self.theme_monitor_timer.start(5000)  # Check every 5 seconds
            else:
                if hasattr(self, 'theme_monitor_timer'):
                    self.theme_monitor_timer.stop()
            
            self.apply_theme()
            self._update_theme_combo_text()
            
            # Apply title bar theming immediately for theme changes
            is_dark = self.get_effective_dark_mode()
            self.apply_dark_title_bar(is_dark)
            
            # Force immediate update and repaint
            self.update()
            self.repaint()
            
            # Also apply to any open dialogs immediately
            for dialog in QtWidgets.QApplication.allWidgets():
                if isinstance(dialog, QtWidgets.QDialog) and dialog.isVisible():
                    try:
                        hwnd = int(dialog.winId())
                        if hwnd:
                            import ctypes
                            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
                            
                            # Try both API versions
                            for attribute in [DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1]:
                                try:
                                    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                                        hwnd,
                                        attribute,
                                        ctypes.byref(ctypes.c_int(1 if is_dark else 0)),
                                        ctypes.sizeof(ctypes.c_int)
                                    )
                                    if result == 0:
                                        dialog.update()
                                        dialog.repaint()
                                        break
                                except:
                                    continue
                        # Force dialog to update its theme styling
                        if hasattr(dialog, 'apply_theme_to_all_widgets'):
                            dialog.apply_theme_to_all_widgets()
                    except:
                        pass
            
            # Force an immediate application repaint for instant visual feedback
            QtWidgets.QApplication.processEvents()
            self.update()
            self.repaint()
            
            # Save the theme preference immediately
            self._save_window_geometry()
            
            logger.info(f'Theme changed to: {self.theme_mode} mode')
    
    def on_logging_level_changed(self, level_text):
        """
        Handle logging level dropdown selection change.
        """
        # Get the logging level value from the combo box
        level_value = self.logging_combo.currentData()
        
        if level_value is not None:
            # Apply the new logging level to the logger
            logger.setLevel(level_value)
            
            # Also update the root logger level if it exists
            root_logger = logging.getLogger()
            root_logger.setLevel(level_value)
            
            # Update all handlers to use the new level
            for handler in logger.handlers:
                handler.setLevel(level_value)
            
            for handler in root_logger.handlers:
                handler.setLevel(level_value)
            
            # Save the logging level preference
            self._save_window_geometry()
            
            # Log the change (this will only show if the level allows it)
            logger.info(f'Logging level changed to: {level_text} ({level_value})')
            logger.debug(f'Debug logging is now {"enabled" if level_value <= logging.DEBUG else "disabled"}')
    
    def show_settings_dialog(self):
        """
        Show the settings dialog window.
        """
        try:
            dialog = SettingsDialog(self)
            
            # Apply title bar theming to the dialog
            is_dark = self.get_effective_dark_mode()
            try:
                hwnd = int(dialog.winId())
                if hwnd:
                    import ctypes
                    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                    DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
                    
                    # Try both API versions
                    for attribute in [DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1]:
                        try:
                            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                                hwnd,
                                attribute,
                                ctypes.byref(ctypes.c_int(1 if is_dark else 0)),
                                ctypes.sizeof(ctypes.c_int)
                            )
                            if result == 0:  # S_OK
                                break
                        except:
                            continue
            except:
                pass  # Ignore title bar theming errors
            
            dialog.exec()
            
        except Exception as e:
            logger.error(f'Error showing settings dialog: {e}')
    
    def _update_theme_combo_text(self):
        """
        Update the theme combo box text based on current theme mode and system state.
        """
        if self.theme_mode == 'system':
            system_dark = self.is_system_dark_theme()
            display_text = f"System ({'Dark' if system_dark else 'Light'})"
            # Update the first item's text to show current system theme
            self.theme_combo.setItemText(0, display_text)
        else:
            # Reset system item text
            self.theme_combo.setItemText(0, 'System')
    
    def init_ui(self):
        """
        Initializes the main UI components and layouts.
        Sets up scenario and window dropdowns, step list, and action buttons.
        """
        # Set compact font
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)

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
        self.btn_delete_scenario = QtWidgets.QPushButton("Delete")
        self.btn_delete_scenario.setToolTip('Delete current scenario')
        self.btn_delete_scenario.clicked.connect(self._log_btn_delete_scenario)
        scenario_btn_group_layout = QtWidgets.QHBoxLayout()
        scenario_btn_group_layout.setSpacing(4)
        scenario_btn_group_layout.setContentsMargins(4, 4, 4, 4)
        scenario_btn_group_layout.addWidget(self.btn_new)
        scenario_btn_group_layout.addWidget(self.btn_rename_scenario)
        scenario_btn_group_layout.addWidget(self.btn_delete_scenario)
        scenario_btn_group_layout.addWidget(self.btn_import)
        scenario_btn_group_layout.addWidget(self.btn_export)
        scenario_group_layout.addLayout(scenario_btn_group_layout)
        scenario_group_box.setLayout(scenario_group_layout)

        # Steps group (QGroupBox with title 'Steps')
        steps_group_box = QtWidgets.QGroupBox('Steps (Up = Higher Priority, Down = Lower Priority)')
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
        self.btn_move_up_step = QtWidgets.QPushButton('Move Up')
        self.btn_move_up_step.setToolTip('Move step up (higher priority)')
        self.btn_move_up_step.clicked.connect(self._log_btn_move_up_step)
        self.btn_move_down_step = QtWidgets.QPushButton('Move Down')
        self.btn_move_down_step.setToolTip('Move step down (lower priority)')
        self.btn_move_down_step.clicked.connect(self._log_btn_move_down_step)
        
        # Set all step buttons to have flexible sizing
        step_buttons = [self.btn_add_step, self.btn_edit_step, self.btn_del_step, 
                       self.btn_rename_step, self.btn_move_up_step, self.btn_move_down_step]
        for btn in step_buttons:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.setMinimumWidth(35)  # Further reduced minimum width for full text
            btn.setMaximumHeight(24)  # Keep height consistent
        
        step_btn_group_layout = QtWidgets.QHBoxLayout()
        step_btn_group_layout.setSpacing(2)  # Reduced spacing
        step_btn_group_layout.setContentsMargins(2, 2, 2, 2)  # Reduced margins
        step_btn_group_layout.addWidget(self.btn_add_step)
        step_btn_group_layout.addWidget(self.btn_edit_step)
        step_btn_group_layout.addWidget(self.btn_rename_step)
        step_btn_group_layout.addWidget(self.btn_del_step)
        step_btn_group_layout.addWidget(self.btn_move_up_step)
        step_btn_group_layout.addWidget(self.btn_move_down_step)
        steps_group_layout.addLayout(step_btn_group_layout)
        steps_group_box.setLayout(steps_group_layout)

        # Start/Stop Combined
        self.btn_start_stop = QtWidgets.QPushButton(f'Start ({self.hotkey.upper()})')
        self.btn_start_stop.setMinimumWidth(80)
        self.btn_start_stop.clicked.connect(self._log_btn_start_stop)

        # State label
        self.state_label = QtWidgets.QLabel('State: Paused')
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Resource usage display
        self.resource_group = QtWidgets.QGroupBox('Resource Usage')
        self.resource_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        resource_layout = QtWidgets.QGridLayout()
        resource_layout.setSpacing(4)
        resource_layout.setContentsMargins(6, 6, 6, 6)

        # Memory usage
        self.memory_label = QtWidgets.QLabel('Memory: --')
        resource_layout.addWidget(self.memory_label, 0, 0)

        # CPU usage
        self.cpu_label = QtWidgets.QLabel('CPU: --')
        resource_layout.addWidget(self.cpu_label, 0, 1)

        # GPU memory info (moved up)
        self.gpu_label = QtWidgets.QLabel('GPU: --')
        resource_layout.addWidget(self.gpu_label, 0, 2)

        # System memory
        self.system_memory_label = QtWidgets.QLabel('System: --')
        resource_layout.addWidget(self.system_memory_label, 1, 0)

        # Cache info
        self.cache_label = QtWidgets.QLabel('Cache: --')
        resource_layout.addWidget(self.cache_label, 1, 1)

        # Performance info (moved down)
        self.performance_label = QtWidgets.QLabel('Performance: --')
        resource_layout.addWidget(self.performance_label, 1, 2)

        self.resource_group.setLayout(resource_layout)

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

        # Resource usage group
        layout.addWidget(self.resource_group, 5, 0, 1, 5)
        layout.setRowStretch(5, 0)  # Resource group should not stretch vertically

        # Start/State row in a group
        start_state_group = QtWidgets.QGroupBox()
        start_state_group.setTitle("")
        start_state_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        start_state_layout = QtWidgets.QHBoxLayout()
        start_state_layout.setSpacing(6)  # Fixed gap between elements
        start_state_layout.addWidget(self.btn_start_stop)
        start_state_layout.addWidget(self.state_label)
        start_state_layout.addStretch(1)  # This pushes settings button to the right
        
        # Settings button on the right side
        self.settings_btn = QtWidgets.QPushButton('Settings')
        self.settings_btn.setMinimumWidth(70)
        self.settings_btn.setMaximumWidth(90)
        self.settings_btn.setToolTip('Open settings window to configure theme, logging, and monitoring options')
        self.settings_btn.clicked.connect(self.show_settings_dialog)
        start_state_layout.addWidget(self.settings_btn)
        
        # Initialize settings that were previously in the UI but now handled by dialog
        self.theme_mode = 'system'  # Default theme mode
        self.logging_level = logging.INFO  # Default logging level
        
        # Create theme and logging combos for settings dialog to reference
        # These are not added to the UI layout but used by the settings dialog
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItem('System', 'system')
        self.theme_combo.addItem('Light', 'light')
        self.theme_combo.addItem('Dark', 'dark')
        self.theme_combo.setCurrentText('System')  # Default to system
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        
        self.logging_combo = QtWidgets.QComboBox()
        logging_levels = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
            ('CRITICAL', logging.CRITICAL)
        ]
        
        for text, level in logging_levels:
            self.logging_combo.addItem(text, level)
        
        # Set default to INFO level (index 1)
        self.logging_combo.setCurrentIndex(1)
        self.logging_combo.currentTextChanged.connect(self.on_logging_level_changed)
        
        # Create update interval combo for settings dialog to reference
        self.update_interval_combo = QtWidgets.QComboBox()
        intervals = [
            ('0.5s', 500),
            ('1s', 1000),
            ('2s', 2000),
            ('3s', 3000),
            ('5s', 5000),
            ('10s', 10000)
        ]
        
        for text, value in intervals:
            self.update_interval_combo.addItem(text, value)
        
        # Set default to 2 seconds (index 2)
        self.update_interval_combo.setCurrentIndex(2)
        self.update_interval_combo.currentIndexChanged.connect(self._on_update_interval_changed)
        
        start_state_group.setLayout(start_state_layout)
        layout.addWidget(start_state_group, 6, 0, 1, 5)
        layout.setRowStretch(6, 0)  # Prevent vertical stretch

        # Set central widget (must be at the end of init_ui)
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        # Apply initial theme and set combo box text
        self.apply_theme()
        self._update_theme_combo_text()
        
        # Apply title bar theming after window is fully initialized
        QtCore.QTimer.singleShot(100, lambda: self.apply_dark_title_bar(self.get_effective_dark_mode()))

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

    def _log_btn_delete_scenario(self):
        """
        Log Delete Scenario button press and call delete_scenario.
        """
        logger.info("UI: Delete Scenario button pressed")
        self.delete_scenario()

    def delete_scenario(self):
        """
        Delete the current scenario after confirmation.
        """
        if not self.current_scenario:
            return
        name = self.current_scenario.name
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Scenario",
            f"Are you sure you want to delete the scenario '{name}'? This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                scenario_dir = self.current_scenario.get_scenario_dir()
                if os.path.exists(scenario_dir):
                    shutil.rmtree(scenario_dir)
                logger.info(f"Deleted scenario: {name}")
                self.current_scenario = None
                self.load_scenarios()
            except Exception as e:
                logger.error(f"Failed to delete scenario '{name}': {e}")
                QtWidgets.QMessageBox.critical(self, "Delete Error", f"Failed to delete scenario.\n{e}")

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

    def _log_btn_move_up_step(self):
        """
        Log Move Up Step button press and call move_up_step.
        """
        logger.info("UI: Move Up Step button pressed")
        self.move_up_step()

    def _log_btn_move_down_step(self):
        """
        Log Move Down Step button press and call move_down_step.
        """
        logger.info("UI: Move Down Step button pressed")
        self.move_down_step()

    def move_up_step(self):
        """
        Move the selected step up in the steps list (higher priority).
        """
        idx = self.selected_step_idx
        if (
            self.current_scenario
            and idx is not None
            and idx > 0
            and idx < len(self.current_scenario.steps)
        ):
            self.current_scenario.steps[idx - 1], self.current_scenario.steps[idx] = (
                self.current_scenario.steps[idx],
                self.current_scenario.steps[idx - 1],
            )
            self.current_scenario.save()
            self.refresh_lists()
            self.steps_list.setCurrentRow(idx - 1)
            self.selected_step_idx = idx - 1

    def move_down_step(self):
        """
        Move the selected step down in the steps list (lower priority).
        """
        idx = self.selected_step_idx
        if (
            self.current_scenario
            and idx is not None
            and idx >= 0
            and idx < len(self.current_scenario.steps) - 1
        ):
            self.current_scenario.steps[idx + 1], self.current_scenario.steps[idx] = (
                self.current_scenario.steps[idx],
                self.current_scenario.steps[idx + 1],
            )
            self.current_scenario.save()
            self.refresh_lists()
            self.steps_list.setCurrentRow(idx + 1)
            self.selected_step_idx = idx + 1

    def select_step(self, idx):
        """
        Set the selected step index in the UI.
        """
        self.selected_step_idx = idx

    def add_step(self):
        """
        Open the dialog to add a new step to the current scenario.
        """
        self._step_dialog = StepDialog(self)
        self._step_dialog.accepted.connect(self._on_add_step_accepted)
        self._step_dialog.show()

    def _on_add_step_accepted(self):
        step = self._step_dialog.get_step()
        self.current_scenario.steps.append(step)
        self.current_scenario.save()
        self.refresh_lists()
        logger.info(f'Added step: {step}')
        self._step_dialog.deleteLater()
        self._step_dialog = None

    def edit_step(self):
        """
        Open the dialog to edit the selected step in the current scenario.
        """
        idx = self.selected_step_idx
        if idx is None or idx < 0 or idx >= len(self.current_scenario.steps):
            return
        step = self.current_scenario.steps[idx]
        self._step_dialog = StepDialog(self, step)
        self._step_dialog.accepted.connect(lambda: self._on_edit_step_accepted(idx))
        self._step_dialog.show()

    def _on_edit_step_accepted(self, idx):
        self.current_scenario.steps[idx] = self._step_dialog.get_step()
        self.current_scenario.save()
        self.refresh_lists()
        logger.info(f'Edited step at idx {idx}')
        self._step_dialog.deleteLater()
        self._step_dialog = None

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
            imgs = ','.join([
                f"{img.get('name', 'img')} ({img.get('sensitivity', 0.9):.2f})"
                for img in step.get('images', [])
            ])
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
        # Images Group
        images_group = QtWidgets.QGroupBox('Images')
        images_group_layout = QtWidgets.QVBoxLayout()
        images_row_layout = QtWidgets.QHBoxLayout()
        img_list_layout = QtWidgets.QVBoxLayout()
        self.img_list = QtWidgets.QListWidget()
        self.img_list.currentItemChanged.connect(self.update_image_preview)
        for img in self.images:
            self.img_list.addItem(img.get('name', 'img'))
        img_list_layout.addWidget(self.img_list)
        img_list_layout.addWidget(QtWidgets.QLabel('Sensitivity:'))
        self.sensitivity_spin = QtWidgets.QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.5, 1.0)
        self.sensitivity_spin.setSingleStep(0.01)
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setValue(0.9)
        self.sensitivity_spin.setToolTip('Sensitivity for selected image (higher = stricter match)')
        self.img_list.currentRowChanged.connect(self.update_sensitivity_spin)
        self.sensitivity_spin.valueChanged.connect(self.save_sensitivity_for_image)
        img_list_layout.addWidget(self.sensitivity_spin)
        images_row_layout.addLayout(img_list_layout)
        self.img_preview = QtWidgets.QLabel('Image Preview')
        self.img_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setFixedSize(200, 200)
        images_row_layout.addWidget(self.img_preview)
        images_group_layout.addLayout(images_row_layout)
        # Image buttons horizontal, spanning full width
        img_btns_layout = QtWidgets.QHBoxLayout()
        self.btn_add_img = QtWidgets.QPushButton('Add Image')
        self.btn_add_img.clicked.connect(self.add_image_to_step)
        self.btn_del_img = QtWidgets.QPushButton('Delete Image')
        self.btn_del_img.clicked.connect(self.delete_image)
        self.btn_rename_img = QtWidgets.QPushButton("Rename Image")
        self.btn_rename_img.clicked.connect(self.rename_image)
        self.btn_retake_img = QtWidgets.QPushButton("Retake Screenshot")
        self.btn_retake_img.clicked.connect(self.retake_screenshot)
        img_btns_layout.addWidget(self.btn_add_img)
        img_btns_layout.addWidget(self.btn_del_img)
        img_btns_layout.addWidget(self.btn_rename_img)
        img_btns_layout.addWidget(self.btn_retake_img)
        images_group_layout.addLayout(img_btns_layout)
        images_group.setLayout(images_group_layout)

        # Actions Group
        actions_group = QtWidgets.QGroupBox('Actions')
        actions_group_layout = QtWidgets.QVBoxLayout()
        self.act_list = QtWidgets.QListWidget()
        for act in self.actions:
            self.act_list.addItem(act.get('type', 'action'))
        actions_group_layout.addWidget(self.act_list)
        act_btns_layout = QtWidgets.QHBoxLayout()
        self.btn_add_act = QtWidgets.QPushButton('Add Action')
        self.btn_add_act.clicked.connect(self.add_action)
        self.btn_del_act = QtWidgets.QPushButton('Delete Action')
        self.btn_del_act.clicked.connect(self.delete_action)
        self.btn_move_up_act = QtWidgets.QPushButton('Move Up')
        self.btn_move_up_act.clicked.connect(self.move_action_up)
        self.btn_move_down_act = QtWidgets.QPushButton('Move Down')
        self.btn_move_down_act.clicked.connect(self.move_action_down)
        act_btns_layout.addWidget(self.btn_add_act)
        act_btns_layout.addWidget(self.btn_del_act)
        act_btns_layout.addWidget(self.btn_move_up_act)
        act_btns_layout.addWidget(self.btn_move_down_act)
        actions_group_layout.addLayout(act_btns_layout)
        actions_group.setLayout(actions_group_layout)

        # Main Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Step Name:'), 0, 0)
        layout.addWidget(self.name_edit, 0, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Condition:'), 1, 0)
        layout.addWidget(self.cond_combo, 1, 1, 1, 2)
        layout.addWidget(images_group, 2, 0, 1, 3)
        layout.addWidget(actions_group, 3, 0, 1, 3)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns, 4, 1, 1, 2)
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
            self.preview_label.setFixedSize(300, 300)
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
        Optimized for memory efficiency.
        """
        main_window = self.parent()
        step_name = self.name_edit.text()
        if not step_name:
            QtWidgets.QMessageBox.warning(self, "Step Name Required", "Please enter a name for the step before adding an image.")
            return

        main_window.hide()
        self.hide()
        time.sleep(0.3)

        cropped_pil_image = None
        rect_coords = None
        try:
            rect_coords = take_screenshot_with_tkinter()
            if rect_coords:
                # Use more memory-efficient screenshot capture
                bbox = (
                    rect_coords['x'], 
                    rect_coords['y'], 
                    rect_coords['x'] + rect_coords['width'], 
                    rect_coords['y'] + rect_coords['height']
                )
                cropped_pil_image = ImageGrab.grab(bbox=bbox)
            logger.debug("StepDialog.add_image_to_step: Screenshot selection finished.")
        except Exception as e:
            logger.error(f"StepDialog.add_image_to_step: Screenshot failed: {e}")
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to take screenshot: {e}')
        finally:
            self.show()
            main_window.show()

        if cropped_pil_image and rect_coords:
            try:
                # Convert PIL to QPixmap more efficiently
                img_np = np.array(cropped_pil_image.convert('RGB'))
                height, width, channel = img_np.shape
                bytes_per_line = 3 * width
                qimage = QtGui.QImage(img_np.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                cropped_pixmap = QtGui.QPixmap.fromImage(qimage)

                name_dialog = self.NameImageDialog(cropped_pixmap, self)
                if name_dialog.exec():
                    name = name_dialog.get_name()
                    if name:
                        scenario_dir = main_window.current_scenario.get_scenario_dir()
                        step_images_dir = os.path.join(scenario_dir, "steps", step_name)
                        if not os.path.exists(step_images_dir):
                            os.makedirs(step_images_dir)

                        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
                        path = os.path.join(step_images_dir, f'{safe_name}.png')

                        # Save with PNG optimization
                        if cropped_pixmap.save(path, "PNG", quality=90):
                            logger.info(f"Screenshot saved to {path}")
                            img_obj = {
                                'path': path, 
                                'region': [rect_coords['x'], rect_coords['y'], rect_coords['width'], rect_coords['height']], 
                                'name': name, 
                                'sensitivity': 0.9
                            }
                            self.images.append(img_obj)
                            self.img_list.addItem(name)
                        else:
                            logger.error(f"Failed to save screenshot to {path}")
                            QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to save the screenshot file.')
                
                # Clean up memory
                del img_np, qimage, cropped_pixmap
                
            finally:
                # Ensure PIL image is cleaned up
                if cropped_pil_image:
                    cropped_pil_image.close()
                    del cropped_pil_image
                gc.collect()

    def update_image_preview(self, current, previous):
        """
        Update the image preview label when a new image is selected.
        Optimized to prevent memory leaks from large pixmaps.
        """
        # Clear previous pixmap to free memory
        if previous:
            self.img_preview.clear()
        
        if current:
            try:
                idx = self.img_list.row(current)
                if idx < 0 or idx >= len(self.images):
                    self.img_preview.setText('Image Preview')
                    return
                
                path = self.images[idx]['path']
                if not os.path.exists(path):
                    self.img_preview.setText('Image Not Found')
                    return
                
                # Load and scale image efficiently
                pixmap = QtGui.QPixmap(path)
                if pixmap.isNull():
                    self.img_preview.setText('Invalid Image')
                    return
                
                # Scale with memory optimization
                preview_size = self.img_preview.size()
                if pixmap.width() > preview_size.width() or pixmap.height() > preview_size.height():
                    scaled_pixmap = pixmap.scaled(
                        preview_size, 
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                    self.img_preview.setPixmap(scaled_pixmap)
                    # Clear original large pixmap
                    del pixmap
                else:
                    self.img_preview.setPixmap(pixmap)
                    
            except Exception as e:
                logger.error(f"Error updating image preview: {e}")
                self.img_preview.setText('Preview Error')
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

        main_window = self.parent()
        main_window.hide()
        self.hide()
        time.sleep(0.3)

        cropped_pil_image = None
        rect_coords = None
        try:
            rect_coords = take_screenshot_with_tkinter()
            if rect_coords:
                cropped_pil_image = ImageGrab.grab(bbox=(
                    rect_coords['x'], 
                    rect_coords['y'], 
                    rect_coords['x'] + rect_coords['width'], 
                    rect_coords['y'] + rect_coords['height']
                ))
            logger.debug("StepDialog.retake_screenshot: Screenshot selection finished.")
        except Exception as e:
            logger.error(f"StepDialog.retake_screenshot: Screenshot failed: {e}")
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to take screenshot: {e}')
        finally:
            self.show()
            main_window.show()

        if cropped_pil_image and rect_coords:
            img_np = np.array(cropped_pil_image.convert('RGB'))
            height, width, channel = img_np.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(img_np.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            cropped_pixmap = QtGui.QPixmap.fromImage(qimage)

            img_obj = self.images[idx]
            path = img_obj['path']

            if cropped_pixmap.save(path, "PNG"):
                logger.info(f"Screenshot retaken and saved to {path}")
                img_obj['region'] = [rect_coords['x'], rect_coords['y'], rect_coords['width'], rect_coords['height']]
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
