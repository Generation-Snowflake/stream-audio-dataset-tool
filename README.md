# Audio Dataset Recording Application

A professional desktop application built with PyQt6 and PyAudio for recording high-quality audio datasets with an intuitive workflow and modern dark mode interface.

![Application Screenshot - Coming Soon]

## Features

- **🎤 Device Selection**: Dropdown menu lists all available audio input devices (ASIO/DirectSound/ALSA)
- **📊 Level Meter**: Real-time audio level visualization with color-coded gradient
- **🧪 Test Mode**: Monitor input levels without recording to verify microphone setup
- **⚙️ Configurable Parameters**: Set recording duration, filename prefix, and starting index
- **✅ Dual Recording Modes**: Separate "Record OK" and "Record NG" buttons for dataset categorization
- **🔢 Auto-Increment**: Automatically increments file index after each recording
- **📁 Organized Output**: Saves files to `output/OK/` or `output/NG/` folders automatically
- **🎨 Dark Mode UI**: Modern, professional dark theme interface
- **🎵 High-Quality Audio**: 48kHz, 16-bit PCM, mono WAV format

## Technical Specifications

### Audio Format

- **Sample Rate**: 48,000 Hz
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono (1 channel)
- **Format**: WAV (WAVE_FORMAT_PCM)
- **Bitrate**: 768 kbps (48kHz × 16-bit × 1 channel)

### File Naming

- Format: `[prefix]_[index].wav`
- Example: `sample_1.wav`, `sample_2.wav`, etc.

### Directory Structure

```
project/
├── audio_recorder.py
├── requirements.txt
├── README.md
└── output/
    ├── OK/
    │   ├── sample_1.wav
    │   ├── sample_2.wav
    │   └── ...
    └── NG/
        ├── sample_10.wav
        ├── sample_11.wav
        └── ...
```

## Installation

### Quick Start (Linux)

```bash
# Run the automated installation script
chmod +x install.sh
./install.sh
```

### Manual Installation

#### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies for PyAudio
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev

# Install Python dependencies
pip install -r requirements.txt
```

### Windows

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### macOS

```bash
# Install portaudio using Homebrew
brew install portaudio

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
python audio_recorder.py
```

### Workflow

1. **Select Audio Device**

   - Choose your microphone from the dropdown menu
   - All available input devices will be listed

2. **Test Your Microphone** (Optional but Recommended)

   - Click "Start Test" button
   - Speak into your microphone
   - Watch the level meter respond to your voice
   - Green (low) → Yellow (medium) → Red (high)
   - Click "Stop Test" when satisfied

3. **Configure Recording Parameters**

   - **Duration**: Set recording length (1-300 seconds)
   - **Prefix**: Enter filename prefix (e.g., "voice", "sample")
   - **Starting Index**: Set initial file number (automatically increments)

4. **Record Your Dataset**

   - Click **"Record OK"** for good recordings → saved to `output/OK/`
   - Click **"Record NG"** for bad/rejected recordings → saved to `output/NG/`
   - The application will:
     - Record for the specified duration
     - Show real-time level meter during recording
     - Save the file automatically
     - Increment the index
     - Display the next filename

5. **Repeat**
   - After each recording, the index automatically increments
   - Ready for the next recording immediately
   - No need to manually manage filenames or folders

### Tips

- Use headphones to monitor playback while recording to avoid feedback
- Maintain consistent distance from the microphone for uniform quality
- The level meter should typically peak in the yellow-green range (50-80%)
- Avoid red peaks (100%) as they may cause clipping/distortion

## Troubleshooting

### "No input devices found!"

- **Linux**: Install `portaudio19-dev` and `python3-pyaudio`
- **Windows**: Install PyAudio: `pip install PyAudio`
- **macOS**: Install portaudio: `brew install portaudio`

### "Failed to start test!"

- Check if another application is using the microphone
- Try selecting a different audio device
- Restart the application
- Check system audio permissions

### Poor audio quality

- Verify sample rate is 48,000 Hz in the code
- Check microphone quality and connection
- Reduce background noise
- Adjust input gain in system settings

### Application crashes on recording

- Ensure sufficient disk space
- Check write permissions for `output/` directory
- Verify PyAudio is correctly installed

## Dependencies

- **PyQt6** (≥6.6.0): Modern GUI framework
- **PyAudio** (≥0.2.13): Audio I/O library
- **NumPy**: Audio signal processing (installed with PyAudio)

## License

This project is open source and available for use in audio dataset creation projects.

## Contributing

Feel free to submit issues or pull requests for improvements:

- Additional audio format support
- Batch processing features
- Waveform visualization
- Audio effects/filters
- Metadata tagging

## Author

Created for professional audio dataset recording workflows.
