"""
Motor Controller - Stepper motor control via Arduino serial communication.
Controls stepper motor through a motor driver with pulse/dir/enable pins.
"""

import time
import threading
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QObject, pyqtSignal


class MotorController(QObject):
    """Controls a stepper motor via Arduino Uno serial port.
    
    Sends text commands to Arduino which drives the motor driver.
    Default wiring: Pin 9 = Pulse, Pin 8 = Dir, Pin 7 = Enable.
    Default microstep: 1600 steps/revolution.
    
    Protocol:
      Python sends: COMMAND\\n
      Arduino replies: READY / OK / DONE / ERROR
    """

    cycle_complete = pyqtSignal()
    status_update = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.serial_conn = None
        self.is_connected = False
        self._is_running = False
        self._serial_lock = threading.Lock()

        # Default configuration
        self.baud_rate = 115200
        self.pulse_pin = 9
        self.dir_pin = 8
        self.enable_pin = 7
        self.microsteps_per_rev = 1600

        # Default motion parameters
        self.rotation_degrees = 180
        self.motor_speed = 800  # steps/sec
        self.pause_after_cycle = 2.0  # seconds

    @staticmethod
    def list_serial_ports():
        """List all available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [
            {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            }
            for port in ports
        ]

    def connect(self, port, baud_rate=None):
        """Connect to Arduino on the specified serial port.
        
        Waits for the Arduino READY handshake, then sends pin configuration.
        Returns True on success.
        """
        if self.is_connected:
            self.disconnect()

        if baud_rate is not None:
            self.baud_rate = baud_rate

        try:
            self.status_update.emit(f"Opening {port}...")
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=5  # generous timeout for Arduino reset
            )

            # Arduino resets when serial opens — wait for READY
            self.status_update.emit("Waiting for Arduino to boot...")
            ready = self._wait_for_response("READY", timeout=5)
            if not ready:
                self.serial_conn.close()
                self.serial_conn = None
                self.error.emit("Arduino did not send READY — check sketch is uploaded")
                return False

            self.status_update.emit("Arduino READY, sending config...")

            # Send pin configuration and verify OK
            if not self._send_config():
                self.serial_conn.close()
                self.serial_conn = None
                self.error.emit("Pin configuration failed")
                return False

            # Verify connection with a PING
            if not self._ping():
                self.serial_conn.close()
                self.serial_conn = None
                self.error.emit("PING failed — communication error")
                return False

            self.is_connected = True
            self.serial_conn.timeout = 2  # normal timeout for commands
            self.status_update.emit(f"Connected to {port}")
            return True

        except serial.SerialException as e:
            self.error.emit(f"Connection failed: {str(e)}")
            if self.serial_conn:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
            self.serial_conn = None
            return False

    def disconnect(self):
        """Disconnect from Arduino."""
        self._is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                with self._serial_lock:
                    self.serial_conn.write(b"STOP\n")
                    self.serial_conn.flush()
                    time.sleep(0.1)
                    self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.is_connected = False
        self.status_update.emit("Disconnected")

    def _wait_for_response(self, expected, timeout=10):
        """Read lines from serial until we get the expected response or timeout."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        
        old_timeout = self.serial_conn.timeout
        self.serial_conn.timeout = timeout
        deadline = time.time() + timeout

        try:
            while time.time() < deadline:
                line = self.serial_conn.readline().decode('utf-8', errors='replace').strip()
                if line == expected:
                    return True
                if line:
                    # Log unexpected lines for debugging
                    print(f"[Motor] Unexpected: '{line}' (waiting for '{expected}')")
            return False
        except serial.SerialException as e:
            self.error.emit(f"Serial read error: {str(e)}")
            return False
        finally:
            self.serial_conn.timeout = old_timeout

    def _send_and_wait(self, command, expected="OK", timeout=60):
        """Send a command and wait for the expected response.
        
        This is the core communication method. Thread-safe via _serial_lock.
        Returns True if the expected response was received.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        
        with self._serial_lock:
            try:
                # Clear any stale data in the input buffer
                self.serial_conn.reset_input_buffer()
                
                # Send command
                self.serial_conn.write(f"{command}\n".encode('utf-8'))
                self.serial_conn.flush()

                # Wait for expected response
                old_timeout = self.serial_conn.timeout
                self.serial_conn.timeout = timeout
                
                deadline = time.time() + timeout
                while time.time() < deadline:
                    line = self.serial_conn.readline().decode('utf-8', errors='replace').strip()
                    if line == expected:
                        self.serial_conn.timeout = old_timeout
                        return True
                    if line == "ERROR":
                        self.serial_conn.timeout = old_timeout
                        self.error.emit(f"Arduino ERROR for: {command}")
                        return False
                    if line:
                        print(f"[Motor] Got '{line}' while waiting for '{expected}'")
                
                self.serial_conn.timeout = old_timeout
                self.error.emit(f"Timeout waiting for '{expected}' (cmd: {command})")
                return False

            except serial.SerialException as e:
                self.error.emit(f"Serial error: {str(e)}")
                return False

    def _send_config(self):
        """Send pin and microstep configuration to Arduino."""
        cmd = f"CONFIG {self.pulse_pin} {self.dir_pin} {self.enable_pin} {self.microsteps_per_rev}"
        return self._send_and_wait(cmd, expected="OK", timeout=5)

    def _ping(self):
        """Verify the connection is alive with a PING command."""
        return self._send_and_wait("PING", expected="PONG", timeout=3)

    def degrees_to_steps(self, degrees):
        """Convert degrees to steps based on microsteps/rev setting."""
        return int((degrees / 360.0) * self.microsteps_per_rev)

    def move_forward(self, degrees=None, speed=None):
        """Move motor forward by specified degrees. Blocks until complete."""
        if not self.is_connected:
            self.error.emit("Not connected to Arduino")
            return False

        deg = degrees if degrees is not None else self.rotation_degrees
        spd = speed if speed is not None else self.motor_speed
        steps = self.degrees_to_steps(deg)

        # Calculate expected duration for timeout
        expected_secs = steps / max(spd, 1)
        timeout = max(expected_secs * 2, 10)  # 2x safety margin, at least 10s

        self.status_update.emit(f"Forward {deg}° ({steps} steps)...")
        result = self._send_and_wait(f"FORWARD {steps} {spd}", expected="DONE", timeout=timeout)
        if result:
            self.status_update.emit(f"Forward {deg}° complete")
        return result

    def move_reverse(self, degrees=None, speed=None):
        """Move motor in reverse by specified degrees. Blocks until complete."""
        if not self.is_connected:
            self.error.emit("Not connected to Arduino")
            return False

        deg = degrees if degrees is not None else self.rotation_degrees
        spd = speed if speed is not None else self.motor_speed
        steps = self.degrees_to_steps(deg)

        expected_secs = steps / max(spd, 1)
        timeout = max(expected_secs * 2, 10)

        self.status_update.emit(f"Reverse {deg}° ({steps} steps)...")
        result = self._send_and_wait(f"REVERSE {steps} {spd}", expected="DONE", timeout=timeout)
        if result:
            self.status_update.emit(f"Reverse {deg}° complete")
        return result

    def stop(self):
        """Emergency stop the motor."""
        self._is_running = False
        if self.is_connected and self.serial_conn and self.serial_conn.is_open:
            try:
                # Don't use _send_and_wait — send raw for speed
                self.serial_conn.write(b"STOP\n")
                self.serial_conn.flush()
                self.status_update.emit("Motor stopped")
            except serial.SerialException:
                pass

    def run_collection_cycle(self, degrees=None, speed=None, pause_sec=None):
        """Run a full collection cycle in a background thread.
        
        Sequence: forward → backward → pause
        Emits cycle_complete when done.
        """
        thread = threading.Thread(
            target=self._collection_cycle_thread,
            args=(degrees, speed, pause_sec),
            daemon=True
        )
        thread.start()

    def _collection_cycle_thread(self, degrees=None, speed=None, pause_sec=None):
        """Thread function for the collection cycle."""
        self._is_running = True

        deg = degrees if degrees is not None else self.rotation_degrees
        spd = speed if speed is not None else self.motor_speed
        pause = pause_sec if pause_sec is not None else self.pause_after_cycle

        try:
            # Step 1: Forward
            if not self._is_running:
                self.cycle_complete.emit()
                return
            self.status_update.emit(f"Cycle: Forward {deg}°...")
            if not self.move_forward(deg, spd):
                self.error.emit("Forward move failed")
                self._is_running = False
                self.cycle_complete.emit()
                return

            # Step 2: Reverse
            if not self._is_running:
                self.cycle_complete.emit()
                return
            self.status_update.emit(f"Cycle: Reverse {deg}°...")
            if not self.move_reverse(deg, spd):
                self.error.emit("Reverse move failed")
                self._is_running = False
                self.cycle_complete.emit()
                return

            # Step 3: Pause
            if not self._is_running:
                self.cycle_complete.emit()
                return
            self.status_update.emit(f"Cycle: Pausing {pause}s...")
            pause_end = time.time() + pause
            while time.time() < pause_end and self._is_running:
                time.sleep(0.1)

            self._is_running = False
            self.status_update.emit("Cycle complete")
            self.cycle_complete.emit()

        except Exception as e:
            self._is_running = False
            self.error.emit(f"Cycle error: {str(e)}")
            self.cycle_complete.emit()

    def update_config(self, pulse_pin=None, dir_pin=None, enable_pin=None, microsteps=None):
        """Update pin configuration and resend to Arduino if connected."""
        if pulse_pin is not None:
            self.pulse_pin = pulse_pin
        if dir_pin is not None:
            self.dir_pin = dir_pin
        if enable_pin is not None:
            self.enable_pin = enable_pin
        if microsteps is not None:
            self.microsteps_per_rev = microsteps

        if self.is_connected:
            self._send_config()
