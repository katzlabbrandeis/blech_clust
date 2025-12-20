# Fixed Time Windows Update

## Summary

Updated the snippet extraction system to use fixed time windows instead of ratio-based scaling, as requested. The snippet window durations are now independent of the main plot window duration.

## Changes Made

### 1. Parameter Changes
**Before:**
```python
self.snippet_window_ratio = 0.2  # 1/5 of main window
```

**After:**
```python
self.snippet_before_ms = 0.5  # Fixed 0.5ms before threshold
self.snippet_after_ms = 1.0   # Fixed 1.0ms after threshold
```

### 2. GUI Control Changes
**Before:**
- Single "Snippet Ratio" text box

**After:**
- "Before (ms)" text box (default: 0.5)
- "After (ms)" text box (default: 1.0)
- Separate controls for precise time window specification

### 3. Extraction Logic Changes
**Before:**
```python
# Ratio-based calculation
snippet_duration_s = self.window_duration * self.snippet_window_ratio
snippet_samples = int(snippet_duration_s * sampling_rate)
before_samples = snippet_samples // 2
after_samples = snippet_samples - before_samples
```

**After:**
```python
# Fixed time window calculation
before_samples = int((self.snippet_before_ms / 1000.0) * sampling_rate)
after_samples = int((self.snippet_after_ms / 1000.0) * sampling_rate)
```

### 4. Callback Method Changes
**Before:**
- `_on_snippet_ratio_change()`

**After:**
- `_on_snippet_before_change()`
- `_on_snippet_after_change()`

## Benefits of Fixed Time Windows

### 1. Consistent Snippet Duration
- Snippet windows remain constant regardless of main plot zoom level
- Easier comparison of waveforms across different viewing windows
- More predictable for spike analysis workflows

### 2. Precise Control
- Users can specify exact millisecond values for before/after windows
- Better alignment with neurophysiology standards (e.g., 1ms after spike peak)
- Independent optimization of snippet windows for different spike types

### 3. Neurophysiology Standards
- Default 0.5ms before / 1.0ms after aligns with common spike analysis practices
- Sufficient time to capture spike waveform shape
- Allows for baseline and recovery period analysis

## Layout Preserved

The current plot layout remains unchanged:
- **Main plot**: 5/6 width (left side)
- **Snippet plot**: 1/6 width (right side, 1/5 of main plot width)
- **Side-by-side arrangement**: Optimal space utilization

## Testing Results

### Automated Tests
- ✅ Default values (0.5ms before, 1.0ms after) correctly set
- ✅ GUI controls properly positioned and functional
- ✅ Snippet extraction produces correct window lengths
- ✅ Parameter changes update extraction correctly
- ✅ All callback methods work as expected

### Validation
- **0.5ms before**: 15 samples at 30kHz sampling rate
- **1.0ms after**: 30 samples at 30kHz sampling rate
- **Total snippet**: 45 samples (1.5ms duration)
- **Parameter changes**: Correctly update snippet lengths

## Usage Examples

### Default Usage
```python
# Default settings automatically applied:
# - 0.5ms before threshold crossing
# - 1.0ms after threshold crossing
# - Up to 50 snippets displayed
```

### Custom Windows
```python
# In GUI:
# 1. Set "Before (ms)" to "0.8"
# 2. Set "After (ms)" to "1.2"
# Result: 0.8ms before, 1.2ms after threshold crossings
```

### Asymmetric Windows
```python
# Example for different spike types:
# Fast spikes: 0.3ms before, 0.7ms after
# Slow spikes: 1.0ms before, 2.0ms after
```

## Implementation Complete

The fixed time window system is now fully implemented and tested:

1. ✅ **Parameters**: Fixed time windows replace ratio-based system
2. ✅ **GUI**: Separate before/after controls with defaults
3. ✅ **Logic**: Extraction uses millisecond-based calculations
4. ✅ **Testing**: Comprehensive validation of functionality
5. ✅ **Documentation**: Updated to reflect new control scheme

The snippet extraction system now provides precise, consistent time windows that are independent of the main plot viewing duration, while maintaining the optimized side-by-side layout for efficient visualization.
