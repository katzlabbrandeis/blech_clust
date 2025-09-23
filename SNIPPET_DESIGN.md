# Waveform Snippet Extraction Design

## Overview
Add waveform snippet extraction and plotting capabilities to the raw data viewer, allowing users to visualize individual spike waveforms that cross the threshold within the current viewing window.

## Design Requirements

### 1. Snippet Extraction Logic
- Extract waveforms that cross the threshold in the current viewing window
- Use similar logic to `extract_waveforms_abu` from `utils/clustering.py`
- Support both positive and negative threshold crossings
- Default snippet windows: 0.5ms before, 1.0ms after threshold crossing

### 2. GUI Controls
- **Snippet Before (ms)**: Text box for time before threshold crossing (default: 0.5)
- **Snippet After (ms)**: Text box for time after threshold crossing (default: 1.0)
- **Show Snippets**: Checkbox to enable/disable snippet extraction and display
- **Max Snippets**: Text box to limit number of snippets displayed (default: 50)

### 3. Display Layout
- **Main Plot**: Current continuous data view (unchanged)
- **Snippet Plot**: New subplot below main plot showing extracted waveforms
- **Snippet Statistics**: Display count of extracted snippets

### 4. Integration Points
- Use existing threshold value from threshold_box
- Extract snippets only from currently visible time window
- Apply same filtering as main display
- Use same data conversion factor (0.6745)

## Technical Implementation

### New Attributes
```python
self.show_snippets = False
self.snippet_before_ms = 0.5  # ms before threshold crossing
self.snippet_after_ms = 1.0   # ms after threshold crossing
self.max_snippets = 50        # maximum snippets to display
self.current_snippets = []    # extracted snippet waveforms
self.snippet_times = []       # times of snippet peaks
```

### New Methods
```python
def _extract_snippets(self, data, time_array, threshold)
def _plot_snippets(self)
def _on_snippet_before_change(self, text)
def _on_snippet_after_change(self, text)
def _on_show_snippets_change(self, label)
def _on_max_snippets_change(self, text)
```

### GUI Layout Changes
- Expand from 6x6 to 7x6 grid
- Add snippet controls in row 6
- Add snippet subplot below main plot

### Snippet Extraction Algorithm
1. Check if threshold is set and show_snippets is enabled
2. Find threshold crossings in current data window
3. For each crossing, extract snippet with specified before/after times
4. Limit to max_snippets (take first N or random sample)
5. Store snippets and their peak times
6. Plot snippets in separate subplot

### Display Features
- Overlay all snippets in snippet subplot
- Color-code by polarity (negative/positive crossings)
- Show snippet count and statistics
- Align snippets by threshold crossing point
- Add average waveform overlay

## User Workflow

1. **Set Threshold**: Enter threshold value in existing threshold box
2. **Enable Snippets**: Check "Show Snippets" checkbox
3. **Configure Parameters**: Adjust snippet before/after times if needed
4. **View Results**: Snippets automatically extracted and displayed
5. **Navigate**: Snippets update as user navigates through data
6. **Filter**: Snippets respect current filter settings

## Performance Considerations

- Only extract snippets from current viewing window (not entire dataset)
- Limit maximum number of snippets to prevent performance issues
- Cache snippets for current window to avoid re-extraction
- Update snippets only when window changes or parameters change

## Error Handling

- Handle cases where no threshold crossings are found
- Ensure snippet windows don't exceed data boundaries
- Gracefully handle invalid parameter values
- Provide user feedback when no snippets are available

## Future Enhancements

- Snippet clustering/classification
- Export snippet data
- Snippet-based measurements (amplitude, width, etc.)
- Real-time snippet statistics
- Snippet template matching