# Dropdown Menu Design for Electrode Selection

## Current System Analysis
- **Current**: RadioButtons widget showing first 5 channels only
- **Limitation**: Can't access channels beyond the first 5
- **Problem**: Datasets may have 16, 32, 64, or more electrodes

## Proposed Solution: Dropdown Menu

### Option 1: Custom Dropdown with Button + Menu
```python
# Create a button that shows current selection
self.channel_button = Button(ax_channel, f'Ch: {self.current_channel}')
self.channel_button.on_clicked(self._show_channel_menu)

# On click, show a popup menu with all channels
def _show_channel_menu(self, event):
    # Create popup menu with all available channels
    # Handle selection and update current channel
```

### Option 2: Text Box with Autocomplete/Validation
```python
# Text box where user can type channel name
self.channel_box = TextBox(ax_channel, 'Channel', initial=self.current_channel)
self.channel_box.on_submit(self._on_channel_text_change)

# Validate input against available channels
def _on_channel_text_change(self, text):
    if text in self.available_channels:
        self.current_channel = text
        self._update_display()
```

### Option 3: Hybrid Button + Text Approach
```python
# Button shows current channel, click cycles through
# Text box allows direct input
self.channel_button = Button(ax_channel, f'Ch: {self.current_channel}')
self.channel_text = TextBox(ax_channel_text, '', initial='')
```

## Recommended Approach: Custom Dropdown

### Implementation Strategy:
1. **Main Button**: Shows current channel selection
2. **Click Handler**: Opens dropdown list when clicked
3. **Dropdown List**: Shows all available channels in scrollable list
4. **Selection Handler**: Updates current channel and closes dropdown
5. **Visual Feedback**: Highlight current selection

### Benefits:
- **Scalable**: Works with any number of electrodes
- **Intuitive**: Familiar dropdown interface
- **Space Efficient**: Takes same space as current radio buttons
- **Complete Access**: All channels accessible, not just first 5

### Technical Implementation:
```python
class DropdownChannelSelector:
    def __init__(self, ax, channels, current_channel, callback):
        self.ax = ax
        self.channels = channels
        self.current_channel = current_channel
        self.callback = callback
        self.dropdown_open = False

        # Main button
        self.button = Button(ax, f'Ch: {current_channel}')
        self.button.on_clicked(self._toggle_dropdown)

        # Dropdown menu (initially hidden)
        self.dropdown_items = []
        self._create_dropdown_items()

    def _toggle_dropdown(self, event):
        if self.dropdown_open:
            self._close_dropdown()
        else:
            self._open_dropdown()

    def _open_dropdown(self):
        # Show dropdown items
        for item in self.dropdown_items:
            item.set_visible(True)
        self.dropdown_open = True

    def _close_dropdown(self):
        # Hide dropdown items
        for item in self.dropdown_items:
            item.set_visible(False)
        self.dropdown_open = False
```

## Alternative: Simple Cycling Button

For immediate implementation, a simpler approach:
- Button shows current channel
- Click cycles to next channel
- Right-click cycles to previous channel
- Text shows "Ch: electrode_name (X/Y)" where X is current, Y is total

This provides:
- **Immediate access** to all channels
- **Simple implementation**
- **Familiar interface** (like channel buttons on oscilloscopes)
- **Space efficient**

## Recommendation

Start with **Simple Cycling Button** for immediate improvement, then enhance to full dropdown if needed.
