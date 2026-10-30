/*
  ThermoGyro logger -- ESP32 + MPU-6050, prints a CSV log over Serial that the
  ThermoGyro web app (Module 3, "Calibrate your own chip") can fit directly.

  STATUS: written for this project but NOT yet run on hardware (no board was
  available before the deadline). It uses only the Wire library and the
  register map from the MPU-6050 datasheet (PWR_MGMT_1 0x6B, CONFIG 0x1A,
  GYRO_CONFIG 0x1B, ACCEL_XOUT_H 0x3B, TEMP_OUT_H 0x41, GYRO_XOUT_H 0x43).

  How to log:
   1. Wire MPU-6050: VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22.
   2. Flash, open Serial Monitor at 115200 baud, keep the board perfectly still.
   3. Slowly warm it (hair dryer from ~30 cm, or carry it from AC room to sun)
      and let it cool again, for 30-60 minutes.
   4. Copy the printed lines into a file log.csv (keep the header line) and
      drop it on the web app. Units are already deg/s and degC.
*/
#include <Wire.h>

const uint8_t MPU = 0x68;
const float GYRO_LSB_PER_DPS = 131.0f;   // +-250 dps full scale (datasheet)
const uint32_t PERIOD_MS = 100;          // 10 Hz; the app averages to 10 s anyway

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}

int16_t read16(uint8_t hi) {
  return (int16_t)((Wire.read() << 8) | Wire.read());
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  writeReg(0x6B, 0x01);   // wake up, clock = PLL with X-gyro reference
  writeReg(0x1A, 0x03);   // DLPF ~44 Hz
  writeReg(0x1B, 0x00);   // gyro +-250 dps
  delay(200);
  Serial.println("t_s,temp_C,gx,gy,gz");
}

void loop() {
  static uint32_t next = millis();
  if ((int32_t)(millis() - next) < 0) return;
  next += PERIOD_MS;

  Wire.beginTransmission(MPU); Wire.write(0x41); Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)8);            // TEMP (2) + GYRO X,Y,Z (6)
  int16_t tRaw = read16(0), gx = read16(0), gy = read16(0), gz = read16(0);

  float tempC = tRaw / 340.0f + 36.53f;          // datasheet formula
  Serial.printf("%.2f,%.3f,%.5f,%.5f,%.5f\n", millis() / 1000.0f, tempC,
                gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS);
}
