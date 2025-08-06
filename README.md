# VisionFlow Automator

VisionFlow Automator is a powerful desktop automation tool that lets you create, edit, and run automation scenarios based on image recognition and GUI actions. With an intuitive graphical interface, you can visually define workflows that interact with any application or window on your system.

## 🌟 Features

- **Scenario Management**: Create, edit, rename, import, export, and organize automation scenarios
- **Template Library**: Get started quickly with pre-built scenario templates
- **Step-Based Automation**: Each scenario consists of multiple steps, each with its own set of images and actions
- **Screenshot-Based Image Matching**: Take region-based screenshots as image triggers for steps with adjustable sensitivity
- **Flexible Actions**: Automate clicks (with position modes), key presses, scrolling, and delays
- **Window Targeting**: Run automations on the entire screen or a specific open window
- **Hotkey Control**: Start/stop automation instantly with a global hotkey (default: F9)
- **Visual Step Editor**: Add, edit, reorder, and preview step images and actions with a modern PyQt6 interface
- **Import/Export**: Share or back up scenarios as ZIP files
- **Persistent State**: Auto-saves all changes and remembers your last scenario and window selection
- **Theme Support**: Light, dark, and system theme modes with instant switching
- **Resource Monitoring**: Real-time performance and memory usage tracking
- **Comprehensive Logging**: Detailed logs for troubleshooting and auditing automation runs

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Required Python packages (see requirements.txt)

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
python main.py
```

## 📋 How It Works

1. **Choose a Template or Create New**: Start with a pre-built template or create your own scenario
2. **Add Step Images**: For each step, take screenshots of UI elements to detect
3. **Define Actions**: Assign actions (click, key, scroll, delay) to perform when images are found
4. **Set Conditions**: Choose whether all images (AND) or any image (OR) must be detected
5. **Target Window**: Select a specific window or use the entire screen
6. **Run Automation**: Start/stop with the hotkey (F9) or UI button

## 📚 Template Library

VisionFlow Automator includes ready-to-use templates in the `/templates/` folder:

### 🖱️ Simple Click Example
Perfect for beginners learning basic automation:
- **Purpose**: Demonstrates basic clicking with image detection
- **Actions**: Left-click at center of detected images with timing delays
- **Use Cases**: Button clicking, basic UI automation, learning the system

### 🌐 Web Browser Example  
Common web browsing automation patterns:
- **Actions**: Page refresh (F5), scrolling, navigation (Alt+Left)
- **Use Cases**: Automated web browsing, content monitoring, web testing
- **Compatible**: Chrome, Firefox, Edge, Safari

### 🎮 Gaming Example
Game automation patterns (use responsibly):
- **Actions**: Attack (Space), abilities (number keys), movement (WASD)
- **Use Cases**: RPG/MMO automation, action games
- **Safety**: Includes warnings about game terms of service

## 🛠️ Advanced Features

### Theme Support
- **Light/Dark/System modes**: Automatic theme switching based on system preferences
- **Instant application**: Themes apply immediately across all windows
- **Persistent settings**: Your theme preference is saved

### Resource Monitoring
- **Real-time metrics**: Memory usage, CPU usage, GPU information
- **Performance tracking**: Loop times, cache statistics
- **Configurable intervals**: Adjust monitoring frequency from 0.5s to 10s

### Window Management
- **Geometry persistence**: Window size and position are remembered
- **Multi-monitor support**: Works across multiple displays
- **Exact positioning**: Precise window placement restoration

## 📝 Requirements

```
PyQt6>=6.0.0
opencv-python>=4.5.0
pyautogui>=0.9.50
pynput>=1.7.0
pygetwindow>=0.0.9
numpy>=1.21.0
Pillow>=8.0.0
```

## 🎯 Tips for Best Results

### Image Recognition
- Use high-quality, tightly-cropped reference images
- Adjust sensitivity per image for robust matching (0.5-1.0 range)
- Test images in the same environment where automation will run
- Avoid images with frequently changing content (timestamps, dynamic text)

### Action Timing
- Add appropriate delays between actions to prevent overwhelming target applications
- Use longer delays after heavy actions (opening applications, loading pages)
- Test with different timing values to find optimal performance

### Window Targeting
- Specify target windows when possible for better performance
- Use "Entire Screen" for cross-application automation
- Refresh window list if target application isn't visible

### Scenario Organization
- Use descriptive names for scenarios and steps
- Group related actions within single steps when logical
- Export important scenarios as backups
- Document complex scenarios with clear step names

## 🔧 Configuration

### Settings Access
Access settings through the "Settings" button in the main window:

- **Theme**: Choose between Light, Dark, or System theme
- **Logging Level**: Adjust verbosity from DEBUG to CRITICAL
- **Resource Monitoring**: Configure update intervals and display options

### File Locations
- **User Scenarios**: Stored in system config directory
- **Templates**: Located in `/templates/` folder within application directory
- **Configuration**: Window geometry and preferences saved automatically
- **Logs**: Application logs for troubleshooting

## 🚦 Safety Guidelines

### General Usage
- Always test automation in safe environments first
- Use appropriate delays to avoid overwhelming target applications
- Monitor automation behavior to ensure it works as expected
- Keep backup copies of important scenarios

### Gaming Automation
- **Check game policies**: Many games prohibit automation tools
- **Respect fair play**: Don't use for unfair competitive advantages  
- **Test safely**: Practice in single-player or private environments
- **Be considerate**: Avoid disrupting other players' experiences

### Application Automation
- **Respect terms of service**: Check if automation is permitted
- **Rate limiting**: Use delays to avoid being detected as bot traffic
- **Error handling**: Monitor for unexpected application behavior
- **Data safety**: Be cautious with automation that handles sensitive information

## 🐛 Troubleshooting

### Common Issues

**Automation not starting:**
- Check that a scenario is selected
- Verify target window is available
- Ensure required dependencies are installed

**Images not detected:**
- Adjust sensitivity settings (try lower values like 0.7-0.8)
- Retake screenshots in the same environment
- Check that target window matches screenshot conditions

**Performance issues:**
- Reduce resource monitoring frequency
- Close unnecessary applications
- Use window targeting instead of full screen
- Optimize scenario step timing

**Application crashes:**
- Check logs for detailed error information
- Verify all dependencies are correctly installed
- Ensure sufficient system memory is available
- Try running with administrator privileges if needed

### Debug Information
- Enable DEBUG logging level for detailed execution information
- Check resource monitoring for performance bottlenecks
- Use step-by-step testing to isolate issues
- Export scenarios before major changes as backups

## 📄 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Bug reports and fixes
- New template scenarios
- Feature enhancements
- Documentation improvements
- Performance optimizations

## 📞 Support

For issues, questions, or feature requests, please use the project's issue tracker.

---

**VisionFlow Automator** — Automate anything you can see! 🎯
