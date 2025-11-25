import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse

def plot_butterworth_responses(order=2):
    """
    Plot frequency response of Butterworth filters:
    1. 300-3000Hz bandpass filter
    2. 300Hz highpass filter
    
    Args:
        order (int): Filter order (default: 2)
    """
    # Sampling frequency (typical for neural data)
    fs = 30000  # Hz
    nyquist = fs / 2
    
    # Filter specifications
    # order is now a parameter
    
    # Bandpass filter: 300-3000 Hz
    low_freq = 300 / nyquist
    high_freq = 3000 / nyquist
    bp_sos = signal.butter(order, [low_freq, high_freq], btype='band', output='sos')
    
    # Highpass filter: 300 Hz
    hp_freq = 300 / nyquist
    hp_sos = signal.butter(order, hp_freq, btype='high', output='sos')
    
    # Frequency range for plotting
    frequencies = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
    
    # Calculate frequency responses
    bp_w, bp_h = signal.sosfreqz(bp_sos, worN=frequencies, fs=fs)
    hp_w, hp_h = signal.sosfreqz(hp_sos, worN=frequencies, fs=fs)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude response
    ax1.semilogx(bp_w, 20 * np.log10(abs(bp_h)), 'b-', label='300-3000 Hz Bandpass', linewidth=2)
    ax1.semilogx(hp_w, 20 * np.log10(abs(hp_h)), 'r-', label='300 Hz Highpass', linewidth=2)
    ax1.set_title(f'Butterworth Filter Frequency Response ({order} Order)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    ax1.set_xlim([10, 10000])
    ax1.set_ylim([-60, 5])
    
    # Add vertical lines at key frequencies
    ax1.axvline(300, color='k', linestyle='--', alpha=0.5, label='300 Hz')
    ax1.axvline(3000, color='k', linestyle=':', alpha=0.5, label='3000 Hz')
    
    # Phase response
    ax2.semilogx(bp_w, np.angle(bp_h) * 180 / np.pi, 'b-', label='300-3000 Hz Bandpass', linewidth=2)
    ax2.semilogx(hp_w, np.angle(hp_h) * 180 / np.pi, 'r-', label='300 Hz Highpass', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    ax2.set_xlim([10, 10000])
    
    # Add vertical lines at key frequencies
    ax2.axvline(300, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(3000, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('butterworth_filter_response.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print filter characteristics
    print("Filter Characteristics:")
    print("=" * 40)
    print(f"Sampling frequency: {fs} Hz")
    print(f"Filter order: {order}")
    print("\nBandpass Filter (300-3000 Hz):")
    print(f"  Lower cutoff: {300} Hz")
    print(f"  Upper cutoff: {3000} Hz")
    print(f"  Normalized frequencies: [{low_freq:.4f}, {high_freq:.4f}]")
    print("\nHighpass Filter (300 Hz):")
    print(f"  Cutoff frequency: {300} Hz")
    print(f"  Normalized frequency: {hp_freq:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Butterworth filter frequency responses')
    parser.add_argument('--order', type=int, default=2, 
                        help='Filter order (default: 2)')
    
    args = parser.parse_args()
    plot_butterworth_responses(order=args.order)
