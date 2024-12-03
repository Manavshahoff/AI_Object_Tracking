#include <Servo.h>
#include <HardwareSerial.h>
#include <Arduino.h>

Servo servoX; // Servo for X-axis rotation
Servo servoY; // Servo for Y-axis rotation

int posX = 90; // Neutral position for the X servo
int posY = 90; // Neutral position for the Y servo

void setup() {
    servoX.attach(9); // Connect servoX to pin 9
    servoY.attach(10); // Connect servoY to pin 10
    Serial.begin(9600); // Start serial communication
    servoX.write(posX);
    servoY.write(posY);
    Serial.println("Servo initialized to center position (90, 90).");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n'); // Read data until newline
        int commaIndex = input.indexOf(',');

        if (commaIndex > 0) {
            // Parse the X and Y offsets from the input
            String offsetXStr = input.substring(0, commaIndex);
            String offsetYStr = input.substring(commaIndex + 1);

            int offsetX = offsetXStr.toInt();
            int offsetY = offsetYStr.toInt();

            // Apply offsets to servo positions
            posX = constrain(posX - offsetX, 0, 180); // Reverse X if needed
            posY = constrain(posY + offsetY, 0, 180); // Adjust Y normally

            // Smoothly move the servos
            servoX.write(posX);
            servoY.write(posY);

            // Debugging output
            Serial.print("ServoX Position: ");
            Serial.print(posX);
            Serial.print(", ServoY Position: ");
            Serial.println(posY);
        }
    }
}
