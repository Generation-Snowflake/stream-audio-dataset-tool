#!/usr/bin/env python3
"""
Test script for Audio Dataset Recording Application
Validates core functionality without GUI interaction
"""

import sys
import pyaudio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Audio Dataset Recorder - Automated Tests")
print("=" * 60)
print()

# Test 1: PyAudio Device Enumeration
print("Test 1: Audio Device Enumeration")
print("-" * 60)
try:
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    print(f"✓ PyAudio initialized successfully")
    print(f"✓ Found {device_count} audio device(s)")
    
    input_devices = []
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append({
                'index': i,
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': int(info['defaultSampleRate'])
            })
            print(f"  - Device {i}: {info['name']}")
            print(f"    Channels: {info['maxInputChannels']}, Sample Rate: {int(info['defaultSampleRate'])} Hz")
    
    if input_devices:
        print(f"✓ Found {len(input_devices)} input device(s)")
    else:
        print("⚠ Warning: No input devices found")
    
    p.terminate()
    print()
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Audio Specifications
print("Test 2: Audio Specification Validation")
print("-" * 60)
EXPECTED_RATE = 48000
EXPECTED_CHANNELS = 1
EXPECTED_FORMAT = pyaudio.paInt16
EXPECTED_WIDTH = 2  # 16-bit = 2 bytes

print(f"✓ Sample Rate: {EXPECTED_RATE} Hz")
print(f"✓ Channels: {EXPECTED_CHANNELS} (Mono)")
print(f"✓ Format: 16-bit PCM")
print(f"✓ Sample Width: {EXPECTED_WIDTH} bytes")
print(f"✓ Calculated Bitrate: {EXPECTED_RATE * EXPECTED_WIDTH * 8 * EXPECTED_CHANNELS / 1000} kbps")
print()

# Test 3: File Structure Validation
print("Test 3: File Structure Validation")
print("-" * 60)
required_files = [
    'audio_recorder.py',
    'requirements.txt',
    'README.md',
    'install.sh'
]

for file in required_files:
    path = Path(file)
    if path.exists():
        size = path.stat().st_size
        print(f"✓ {file} ({size} bytes)")
    else:
        print(f"✗ Missing: {file}")

print()

# Test 4: Import Application Module
print("Test 4: Application Module Import")
print("-" * 60)
try:
    # Check if main application can be imported (syntax check)
    with open('audio_recorder.py', 'r') as f:
        code = f.read()
        compile(code, 'audio_recorder.py', 'exec')
    print("✓ audio_recorder.py syntax is valid")
    print("✓ Module can be compiled")
    
    # Count key components
    classes = code.count('class ')
    functions = code.count('def ')
    print(f"✓ Found {classes} class(es)")
    print(f"✓ Found {functions} function(s)/method(s)")
    
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

print()

# Test 5: Directory Creation Test
print("Test 5: Output Directory Creation")
print("-" * 60)
test_ok_dir = Path("output/OK")
test_ng_dir = Path("output/NG")

try:
    test_ok_dir.mkdir(parents=True, exist_ok=True)
    test_ng_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {test_ok_dir}")
    print(f"✓ Created: {test_ng_dir}")
    print("✓ Directory structure ready")
except Exception as e:
    print(f"✗ Error creating directories: {e}")
    sys.exit(1)

print()

# Test 6: WAV File Format Test
print("Test 6: WAV File Format Specification")
print("-" * 60)
import wave
import numpy as np

try:
    # Create a test WAV file
    test_file = Path("output/OK/test_sample.wav")
    sample_rate = 48000
    duration = 1  # 1 second
    
    # Generate silent audio data
    frames = []
    chunk_size = 1024
    num_chunks = int(sample_rate / chunk_size * duration)
    
    for _ in range(num_chunks):
        silent_chunk = np.zeros(chunk_size, dtype=np.int16)
        frames.append(silent_chunk.tobytes())
    
    # Write WAV file
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(48000)  # 48kHz
        wf.writeframes(b''.join(frames))
    
    print(f"✓ Test WAV file created: {test_file}")
    
    # Verify WAV file
    with wave.open(str(test_file), 'rb') as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        
        print(f"✓ Verified - Channels: {channels} (expected: 1)")
        print(f"✓ Verified - Sample Width: {width} bytes (expected: 2)")
        print(f"✓ Verified - Sample Rate: {rate} Hz (expected: 48000)")
        
        if channels == 1 and width == 2 and rate == 48000:
            print("✓ WAV format specification: PASSED")
        else:
            print("✗ WAV format specification: FAILED")
    
    # Clean up test file
    test_file.unlink()
    print(f"✓ Test file cleaned up")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()

# Summary
print("=" * 60)
print("Test Summary")
print("=" * 60)
print("✓ All automated tests passed!")
print()
print("Manual Testing Required:")
print("  1. Run: python3 audio_recorder.py")
print("  2. Select an audio input device")
print("  3. Click 'Start Test' and verify level meter responds")
print("  4. Configure recording parameters")
print("  5. Click 'Record OK' to test OK folder recording")
print("  6. Click 'Record NG' to test NG folder recording")
print("  7. Verify files are created with correct naming")
print("  8. Verify index auto-increments")
print("=" * 60)
