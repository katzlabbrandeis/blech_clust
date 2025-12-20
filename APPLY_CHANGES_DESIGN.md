# Apply Changes Button Design

## Problem
Current parameter changes trigger immediate `_update_display()` calls, which are slow because they involve:
1. Data loading from HDF5 files
2. Signal filtering operations
3. Snippet extraction and analysis
4. Plot rendering and updates

This makes the UI unresponsive when users adjust multiple parameters.

## Solution
Add an "Apply Changes" button that batches parameter updates and defers expensive operations.

## Design

### 1. Deferred Update System
- Add `auto_update` flag (default: True for backward compatibility)
- When `auto_update = False`, parameter callbacks store values but don't call `_update_display()`
- "Apply Changes" button triggers single `_update_display()` call with all accumulated changes

### 2. Parameters to Defer (Non-Time Related)
- **Filtering**: Lowpass/Highpass frequency changes
- **Thresholds**: Threshold value changes
- **Y-Limits**: Manual Y-axis scaling
- **Snippets**: Before/After times, Max snippets
- **Display**: Data conversion factor

### 3. Parameters to Keep Immediate (Time Related)
- **Time Navigation**: Time slider, Prev/Next buttons, Jump
- **Channel Selection**: Radio button changes (these are fast)
- **Window Duration**: Changes viewing window (excluded per requirements)

### 4. GUI Layout
- Add "Apply Changes" button in control area
- Add "Auto Update" checkbox to toggle between modes
- Visual indicator when changes are pending

### 5. Implementation Strategy
```python
class InteractivePlotter:
    def __init__(self):
        self.auto_update = True  # Default to current behavior
        self.pending_changes = False  # Track if changes need applying

    def _defer_update_if_needed(self):
        """Call _update_display() only if auto_update is enabled."""
        if self.auto_update:
            self._update_display()
        else:
            self.pending_changes = True
            self._update_apply_button_state()

    def _on_apply_changes(self, event):
        """Apply all pending changes."""
        self._update_display()
        self.pending_changes = False
        self._update_apply_button_state()
```

### 6. User Experience
- **Default Mode**: Auto-update enabled, works as before
- **Batch Mode**: User unchecks "Auto Update", makes multiple changes, clicks "Apply Changes"
- **Visual Feedback**: Apply button highlights when changes are pending
- **Flexibility**: User can toggle between modes as needed

### 7. Benefits
- **Performance**: Eliminates redundant expensive operations
- **Responsiveness**: UI remains responsive during parameter adjustment
- **Flexibility**: Users can choose between immediate feedback and batch updates
- **Backward Compatibility**: Default behavior unchanged
