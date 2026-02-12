/*
 * Motor Control Sketch for Audio Dataset Tool
 * 
 * Controls a stepper motor via pulse/dir/enable pins.
 * Receives commands over Serial from the Python application.
 * 
 * Default wiring:
 *   Pin 9 = PULSE (PUL/STEP)
 *   Pin 8 = DIR (Direction)
 *   Pin 7 = EN  (Enable, active LOW)
 * 
 * Motor driver: microstep 1600 steps/revolution
 * 
 * Serial commands (115200 baud):
 *   FORWARD <steps> <speed>   - Move forward, speed in steps/sec
 *   REVERSE <steps> <speed>   - Move reverse, speed in steps/sec
 *   CONFIG <pulse> <dir> <en> <microsteps> - Reconfigure pins
 *   STOP                      - Emergency stop
 *   PING                      - Connection check (replies PONG)
 * 
 * Responses:
 *   READY - Sent on boot
 *   PONG  - Reply to PING
 *   OK    - Command acknowledged (CONFIG, STOP)
 *   DONE  - Movement complete (FORWARD/REVERSE)
 *   ERROR - Invalid command or parameters
 */

// Default pin assignments
int pulsePin = 9;
int dirPin   = 8;
int enPin    = 7;
int microstepsPerRev = 1600;

// State
volatile bool stopRequested = false;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  
  pinMode(pulsePin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  
  // Enable motor driver (active LOW)
  digitalWrite(enPin, LOW);
  digitalWrite(pulsePin, LOW);
  digitalWrite(dirPin, LOW);
  
  // Wait a moment for serial to stabilize, then send READY
  delay(200);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("FORWARD")) {
    handleMove(cmd, HIGH);
  } 
  else if (cmd.startsWith("REVERSE")) {
    handleMove(cmd, LOW);
  } 
  else if (cmd.startsWith("CONFIG")) {
    handleConfig(cmd);
  } 
  else if (cmd == "STOP") {
    handleStop();
  }
  else if (cmd == "PING") {
    Serial.println("PONG");
  }
  else {
    Serial.println("ERROR");
  }
}

void handleMove(String cmd, int direction) {
  // Parse: FORWARD/REVERSE <steps> <speed>
  int firstSpace = cmd.indexOf(' ');
  if (firstSpace < 0) {
    Serial.println("ERROR");
    return;
  }
  
  String params = cmd.substring(firstSpace + 1);
  int secondSpace = params.indexOf(' ');
  if (secondSpace < 0) {
    Serial.println("ERROR");
    return;
  }
  
  long steps = params.substring(0, secondSpace).toInt();
  long speed = params.substring(secondSpace + 1).toInt();
  
  if (steps <= 0 || speed <= 0) {
    Serial.println("ERROR");
    return;
  }
  
  // Calculate delay between pulses (microseconds)
  // speed is steps/sec, so delay = 1,000,000 / (2 * speed) for half-period
  long pulseDelay = 500000L / speed;
  if (pulseDelay < 10) pulseDelay = 10;  // Minimum 10us for safety
  
  // Set direction
  digitalWrite(dirPin, direction);
  delayMicroseconds(10);  // Direction setup time
  
  // Enable motor
  digitalWrite(enPin, LOW);
  
  stopRequested = false;
  
  // Execute steps
  // Check for STOP every N steps to avoid blocking too long
  // For fast speeds, check less frequently to avoid timing jitter
  long checkInterval = max(steps / 20, 10L);  // check ~20 times during move
  
  for (long i = 0; i < steps; i++) {
    if (stopRequested) {
      break;
    }
    
    // Check for incoming STOP command periodically (not every step)
    if (i % checkInterval == 0 && Serial.available() > 0) {
      String check = Serial.readStringUntil('\n');
      check.trim();
      if (check == "STOP") {
        stopRequested = true;
        break;
      }
    }
    
    digitalWrite(pulsePin, HIGH);
    delayMicroseconds(pulseDelay);
    digitalWrite(pulsePin, LOW);
    delayMicroseconds(pulseDelay);
  }
  
  Serial.println("DONE");
}

void handleConfig(String cmd) {
  // Parse: CONFIG <pulsePin> <dirPin> <enPin> <microsteps>
  int idx = cmd.indexOf(' ');
  if (idx < 0) {
    Serial.println("ERROR");
    return;
  }
  
  String params = cmd.substring(idx + 1);
  
  // Parse 4 values
  int values[4];
  
  for (int i = 0; i < 4; i++) {
    params.trim();
    int spaceIdx = params.indexOf(' ');
    
    if (i < 3 && spaceIdx < 0) {
      Serial.println("ERROR");
      return;
    }
    
    if (spaceIdx >= 0) {
      values[i] = params.substring(0, spaceIdx).toInt();
      params = params.substring(spaceIdx + 1);
    } else {
      values[i] = params.toInt();
    }
  }
  
  // Validate pin numbers (2-13 for Uno)
  for (int i = 0; i < 3; i++) {
    if (values[i] < 2 || values[i] > 13) {
      Serial.println("ERROR");
      return;
    }
  }
  
  // Validate microsteps (must be > 0)
  if (values[3] <= 0) {
    Serial.println("ERROR");
    return;
  }
  
  // Reconfigure pins - restore old pins to INPUT first
  pinMode(pulsePin, INPUT);
  pinMode(dirPin, INPUT);
  pinMode(enPin, INPUT);
  
  // Set new pins
  pulsePin = values[0];
  dirPin   = values[1];
  enPin    = values[2];
  microstepsPerRev = values[3];
  
  pinMode(pulsePin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  
  digitalWrite(pulsePin, LOW);
  digitalWrite(dirPin, LOW);
  digitalWrite(enPin, LOW);  // Enable (active LOW)
  
  Serial.println("OK");
}

void handleStop() {
  stopRequested = true;
  digitalWrite(pulsePin, LOW);
  Serial.println("OK");
}
