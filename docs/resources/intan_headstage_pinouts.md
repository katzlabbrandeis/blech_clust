# Intan Headstage Pinouts

This document provides reference information for Intan headstage electrode connector pinouts, which is useful when creating the `electrode_layout_frame` in `blech_exp_info.py`.

## Official Intan Resources

For complete and up-to-date information about Intan headstage pinouts, please refer to the official Intan Technologies documentation:

- **[Intan RHD Headstages](https://intantech.com/RHD_headstages.html)**
- **[Intan RHS Headstages](https://intantech.com/RHS_headstages.html)**
- **[Intan Electrode Adapters](https://intantech.com/electrode_adapters.html)**

## RHD 32-Channel Headstage

The RHD 32-channel headstages have 36-pin electrode connectors that mate with many commercially-available electrodes.

### Key Specifications

- **Connector Type**: 36-pin electrode connector
- **Channels**: 32 amplifier channels
- **Reference**: Configurable via zero-ohm jumper R0 to disconnect reference electrode from ground
- **Available Variants**:
  - Standard RHD 32ch headstage
  - RHD 32ch headstage with accelerometer
  - RHD 32-channel headstage with SPI interface cable

### Pinout Notes

- Two redundant REF pins and two redundant GND pins are provided on the electrode connector
- Only one REF and one GND connection is needed for the entire headstage
- The headstage mates with electrodes from various manufacturers and Intan custom electrode adapters

## RHD 64-Channel Headstage

The RHD 64-channel headstages provide higher density recordings.

### Key Specifications

- **Connector Type**: 36-pin electrode connector (dual connectors for 64 channels)
- **Channels**: 64 amplifier channels
- **Reference**: Configurable via zero-ohm jumper R0
- **Available Variants**:
  - RHD 64ch headstage with ports labeled

### Pinout Notes

- Two redundant REF pins and two redundant GND pins are provided on each electrode connector
- Only one REF and one GND connection is needed per connector

## RHD 16-Channel Bipolar-Input Headstage

Designed for bipolar EMG or other differential recordings.

### Key Specifications

- **Connector Type**: 36-pin electrode connector
- **Channels**: 16 bipolar input channels
- **Reference**: Four redundant GND pins provided

## Using This Information with blech_clust

When setting up experiments with `blech_exp_info.py`, you'll need to specify the electrode layout using the CAR (Common Average Referencing) groups. Understanding the headstage pinout helps you:

1. **Identify channel numbers**: Map the electrode connector pins to specific amplifier channel numbers
2. **Configure CAR groups**: Group electrodes for common average referencing based on their physical layout
3. **Set up EMG channels**: If using integrated EMG channels on the headstage

### Example: Creating Electrode Layout

```bash
# Run experiment info setup
python blech_exp_info.py /path/to/data

# This will generate an electrode_layout.csv file
# that you can edit to specify CAR groups
```

The electrode layout file contains:
- `filename`: The electrode data file
- `electrode_ind`: The electrode index
- `port`: The port on the headstage
- `electrode_num`: The electrode number
- `CAR_group`: The group for common average referencing

## Additional Resources

- [Intan Products Overview](https://intantech.com/products_RHD2000.html)
- [Intan Recording Controllers](https://intantech.com/recording_controller.html)
- [Intan SPI Cables](https://intantech.com/RHD_SPI_cables.html)
