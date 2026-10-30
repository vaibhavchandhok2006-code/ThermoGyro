/*
  ThermoGyro compensation -- ESP32 + MPU-6050 heading with temperature-drift
  compensation and start-up anchoring.

  STATUS: written for this project but NOT yet run on hardware.

  1. Log your own chip with thermogyro_logger and fit it in the web app.
  2. Replace TG_CAL below with the block the app generates (the values here are
     the app's example fit for Cranfield IMU 2 -- a DIFFERENT chip; they will
     not be correct for your sensor, every MPU-6050 drifts differently).
  3. At power-on keep the board still for 3 s (start-up anchoring, measures
     b0 at temperature T0). After that ThermoGyro removes the temperature drift.
*/
#include <Wire.h>

struct TGAxis { float Tm, Ts, W[3]; };
// ---- paste the block from the web app here ----
const TGAxis TG_CAL[3] = {
  {20.40939f, 32.68828f, {-0.5403511f, 0.6153688f, -0.1405400f}},   // gx (example: Cranfield IMU 2)
  {20.40939f, 32.68828f, {-1.332089f, 0.1918704f, -0.03929479f}},   // gy
  {20.40939f, 32.68828f, {-2.587306f, 0.6652648f, 0.08029898f}},    // gz
};
// -----------------------------------------------

const uint8_t MPU = 0x68;
float b0[3] = {0, 0, 0}, T0 = 25.0f, heading = 0, headingRaw = 0;
uint32_t lastUs;

float tg_curve(int axis, float T) {
  const TGAxis &c = TG_CAL[axis]; float z = (T - c.Tm) / c.Ts;
  return c.W[0] + c.W[1] * z + c.W[2] * z * z;
}

void readMPU(float &T, float g[3]) {
  Wire.beginTransmission(MPU); Wire.write(0x41); Wire.endTransmission(false);
  Wire.requestFrom(MPU, (uint8_t)8);
  int16_t t = (Wire.read() << 8) | Wire.read();
  for (int i = 0; i < 3; i++) { int16_t v = (Wire.read() << 8) | Wire.read(); g[i] = v / 131.0f; }
  T = t / 340.0f + 36.53f;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); Wire.setClock(400000);
  Wire.beginTransmission(MPU); Wire.write(0x6B); Wire.write(0x01); Wire.endTransmission();
  Wire.beginTransmission(MPU); Wire.write(0x1B); Wire.write(0x00); Wire.endTransmission();
  delay(200);
  // start-up anchoring: 3 s still
  float sum[3] = {0, 0, 0}, sT = 0, T, g[3]; int n = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < 3000) { readMPU(T, g); for (int i = 0; i < 3; i++) sum[i] += g[i]; sT += T; n++; delay(5); }
  for (int i = 0; i < 3; i++) b0[i] = sum[i] / n;
  T0 = sT / n;
  lastUs = micros();
  Serial.println("t_s,temp_C,heading_startup_only_deg,heading_thermogyro_deg");
}

void loop() {
  float T, g[3]; readMPU(T, g);
  uint32_t now = micros(); float dt = (now - lastUs) * 1e-6f; lastUs = now;
  float gzComp = g[2] - (b0[2] + tg_curve(2, T) - tg_curve(2, T0));
  headingRaw += (g[2] - b0[2]) * dt;
  heading += gzComp * dt;
  static uint32_t lastPrint = 0;
  if (millis() - lastPrint >= 1000) {
    lastPrint = millis();
    Serial.printf("%.1f,%.2f,%.2f,%.2f\n", millis() / 1000.0f, T, headingRaw, heading);
  }
  delay(5);
}
