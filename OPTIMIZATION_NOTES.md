# VisionFlow Automator - Performance Optimizations

## Overview
This document outlines the performance and memory optimizations implemented to prevent the application from being killed or stalled during scenario execution.

## Key Optimizations Implemented

### 1. Memory Management
- **Template Caching**: Added `TemplateCache` class to cache loaded CV2 templates instead of reloading from disk every loop iteration
- **Screenshot Caching**: Implemented screenshot caching with 50ms duration to reduce memory allocations
- **Resource Cleanup**: Added proper cleanup in `cleanup_resources()` method and `closeEvent()`
- **Garbage Collection**: Strategic `gc.collect()` calls to force memory cleanup at appropriate times
- **Pixmap Memory Management**: Optimized image preview handling to prevent large pixmap memory leaks

### 2. Performance Improvements
- **Optimized Loop Structure**: Redesigned automation loop to be more efficient
- **Step Cooldown System**: Prevents rapid re-execution of the same step (1-second cooldown)
- **Priority Execution**: Only executes one step per loop iteration to prevent blocking
- **Dynamic Sleep**: Adjusts sleep time based on loop performance (0.05-0.2s)
- **Performance Monitoring**: Added loop time tracking and warnings for slow performance

### 3. Threading Improvements
- **Proper Thread Management**: Added timeout for thread joining in `stop_automation()`
- **Background Processing**: Improved worker thread handling with better error recovery
- **Non-blocking Operations**: Reduced blocking operations in the main thread

### 4. UI Optimizations
- **Reduced Logging**: Changed default log level from DEBUG to INFO to reduce I/O overhead
- **Efficient Image Previews**: Optimized image scaling and memory usage in dialogs
- **State Update Throttling**: Reduced frequency of UI state updates

### 5. Resource Monitoring
- **Performance Timer**: Added 10-second monitoring timer to track application health
- **Memory Usage Tracking**: Optional psutil integration for detailed memory monitoring
- **Cache Size Monitoring**: Tracks template cache size and warns when it grows large

### 6. Error Handling
- **Graceful Degradation**: Better error handling in screenshot capture and template matching
- **Resource Recovery**: Automatic cleanup on errors to prevent resource leaks
- **Hotkey Error Handling**: Continues operation even if hotkey setup fails

## Configuration Changes

### PyAutoGUI Optimizations
- Set `pyautogui.PAUSE = 0.01` (reduced from default 0.1s)
- Set `pyautogui.MINIMUM_DURATION = 0` for faster actions
- Kept `pyautogui.FAILSAFE = False` for automation reliability

### Screenshot Optimization
- Added minimum size validation (10x10 pixels)
- Improved tkinter screenshot tool with better UX
- Memory-efficient PIL image handling with proper cleanup

### Template Matching
- LRU cache with automatic cleanup of old entries
- Maximum cache size of 50 templates
- Force reload option for template updates

## Performance Targets

### Memory Usage
- Template cache limited to 50 entries
- Screenshot cache duration: 50ms
- Automatic cleanup every 100 loop iterations
- Garbage collection after major operations

### Timing
- Target loop time: 50-200ms
- Step cooldown: 1 second
- Performance warning threshold: 1 second average loop time
- Monitoring interval: 10 seconds

### Thread Safety
- Worker thread timeout: 2 seconds
- Daemon threads for automatic cleanup
- Proper resource locking where needed

## Usage Recommendations

1. **Monitor Performance**: Check logs for performance warnings
2. **Resource Management**: Regularly restart long-running sessions
3. **Template Optimization**: Use appropriately sized template images
4. **Step Design**: Avoid too many simultaneous image detections
5. **System Resources**: Ensure adequate RAM for screenshot operations

## Future Improvements

1. **Multi-threading**: Consider separate threads for image processing
2. **Image Compression**: Compress cached templates to save memory
3. **Region Optimization**: Use smaller detection regions when possible
4. **GPU Acceleration**: Consider OpenCV GPU operations for template matching
5. **Background Processing**: Process non-critical operations in background

## Troubleshooting

### High Memory Usage
- Check template cache size in logs
- Reduce number of simultaneous steps
- Restart application periodically

### Slow Performance
- Check average loop times in logs
- Reduce image template sizes
- Simplify detection regions
- Consider fewer simultaneous detections

### Application Crashes
- Check log files for error patterns
- Monitor system memory usage
- Verify image file integrity
- Check for corrupted templates
