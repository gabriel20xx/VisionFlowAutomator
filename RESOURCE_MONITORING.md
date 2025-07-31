# Resource Usage Monitoring - VisionFlow Automator

## Overview
The VisionFlow Automator now includes real-time resource usage monitoring displayed directly in the UI. This feature helps users monitor the application's performance and system impact.

## Features Added

### 🔍 **Resource Usage Display**
A new "Resource Usage" group box has been added to the main UI showing:

#### Memory Usage
- **Process Memory**: Shows the application's memory usage in MB and percentage of system memory
- **Color-coded warnings**: Green for normal usage, red for high usage (>10% of system memory)

#### CPU Usage
- **Process CPU**: Shows the application's CPU usage percentage
- **Color-coded warnings**: Green for normal usage, red for high usage (>50%)

#### System Information
- **System Memory**: Shows overall system memory usage percentage and available memory in GB
- **Color-coded warnings**: Green for normal, red when system memory usage exceeds 85%

#### Cache Information
- **Template Cache**: Number of cached CV2 templates
- **Cooldown Entries**: Number of active step cooldowns
- **Color-coded status**: Blue for normal, orange when template cache exceeds 20 entries

#### Performance Metrics
- **Loop Time**: Average automation loop execution time in milliseconds
- **Color-coded performance**: Green for fast loops (<500ms), red for slow loops (>500ms)

### 🎛️ **Controls**

#### Toggle Button
- **Hide/Show Resources**: Button to toggle the visibility of resource usage information
- **Space-saving**: Users can hide the resource display if they don't need it
- **Tooltip**: Provides information about psutil installation for enhanced monitoring

#### Interactive Elements
- **Clickable Labels**: When psutil is not installed, clicking on resource labels shows installation instructions
- **Installation Guide**: Built-in dialog explaining how to install psutil for enhanced monitoring

### 📊 **Monitoring Levels**

#### Basic Monitoring (Without psutil)
When psutil is not installed, the display shows:
- Template cache size
- Cooldown entries count
- Basic performance metrics
- "N/A" for system metrics with installation instructions

#### Enhanced Monitoring (With psutil)
When psutil is installed (`pip install psutil`), the display shows:
- Detailed process memory usage
- CPU usage percentage
- System memory statistics
- Available system memory
- Process performance metrics

### ⚙️ **Technical Details**

#### Update Frequency
- **Real-time Updates**: Resource display updates every 2 seconds
- **Responsive UI**: Non-blocking updates that don't interfere with automation
- **Performance Optimized**: Minimal overhead from monitoring

#### Color Coding System
- **🟢 Green**: Normal/Good performance
- **🟡 Orange/Yellow**: Warning levels
- **🔴 Red**: High usage/Performance issues
- **⚫ Gray**: Unavailable/Error states

#### Error Handling
- **Graceful Degradation**: Shows basic info if detailed monitoring fails
- **Error Display**: Shows error messages for troubleshooting
- **Fallback Information**: Always shows cache and performance data

### 🎯 **Benefits**

#### For Users
- **Real-time Monitoring**: See exactly how much resources the app is using
- **Performance Insights**: Identify when the app is running slowly
- **System Impact**: Understand the app's impact on overall system performance
- **Troubleshooting**: Quickly identify memory leaks or performance issues

#### For Developers
- **Performance Metrics**: Built-in performance profiling
- **Memory Leak Detection**: Monitor memory usage over time
- **Cache Optimization**: Track template cache efficiency
- **System Resources**: Monitor system-wide impact

### 📋 **Usage Instructions**

#### Basic Usage
1. **View Resources**: Resource usage is displayed by default in the main window
2. **Toggle Display**: Click "Hide Resources" to save screen space
3. **Monitor Performance**: Watch for color changes indicating performance issues

#### Enhanced Monitoring Setup
1. **Install psutil**: Run `pip install psutil` in your Python environment
2. **Restart Application**: Close and reopen VisionFlow Automator
3. **Enhanced Display**: All metrics will now show detailed system information

#### Interpreting Metrics
- **Memory < 100MB**: Normal usage for typical scenarios
- **CPU < 20%**: Normal usage during automation
- **Loop Time < 200ms**: Good performance
- **Cache < 20 entries**: Normal template usage

### 🚨 **Warning Thresholds**

#### Memory Warnings
- **Process Memory > 10%**: High memory usage warning
- **System Memory > 85%**: System memory critical

#### Performance Warnings  
- **CPU > 50%**: High CPU usage
- **Loop Time > 500ms**: Performance degradation
- **Cache > 20 entries**: Large template cache

### 🔧 **Troubleshooting**

#### High Memory Usage
- Check for memory leaks in long-running sessions
- Clear template cache periodically
- Reduce number of simultaneous image templates

#### High CPU Usage
- Reduce automation frequency
- Optimize image template sizes
- Check for infinite loops in scenarios

#### Slow Performance
- Monitor loop times for bottlenecks
- Reduce screenshot frequency
- Optimize template matching regions

#### Missing psutil
- Install with: `pip install psutil`
- Click on gray resource labels for installation instructions
- Restart application after installation

The resource monitoring feature provides valuable insights into the application's performance and helps users optimize their automation scenarios for better efficiency.
