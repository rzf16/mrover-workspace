Code for rover localization
 ---
This package is maintained by the Localization sub-team and intended for use in the autonomous traversal task.
### About
The `jetson/filter` package runs on the onboard Jetson TX2 and enhances rover localization and state estimation via sensor fusion. Currently, the sensors used are the ZED-F9P GPS and the ICM 20948 IMU. For further information on these sensors and how to use them, see `jetson/gps` and `beaglebone/imu`, respectively.

### Interfaces
**`jetson/gps -> jetson/filter`**
Publisher: `jetson/gps`
Subscriber: `jetson/filter`
Message: `GPS.lcm`
Channel: `/gps`
Description: Outputs of the GPS unit, including coordinates, grounds speed, bearing, and fix quality

**`beaglebone/imu -> jetson/filter`**
Publisher: `beaglebone/imu`
Subscriber: `jetson/filter`
Message: `IMUData.lcm`
Channel: `/imu_data`
Description: Outputs of the IMU, including accelerometer, magnetometer, and gyroscope readings, RPY, and bearing

**`jetson/nav -> jetson/filter`**
Publisher: `jetson/nav`
Subscriber: `jetson/filter`
Message: `NavStatus.lcm`
Channel: `/nav_status`
Description: Current state of the Navigation FSM

**`jetson/filter -> jetson/nav`**
Publisher: `jetson/filter`
Subscriber: `jetson/nav`
Message: `Odometry.lcm`
Channel: `/odometry`
Description: Current state estimate for the rover, including coordinates, speed, and bearing

### Configuration

The configuration file for `jetson/filter` can be found in `config/filter/config.json`. Parameters include:
* FilterType: The type of sensor fusion filter to use. Options are:
  * LKF: Linear Kalman Filter
  * Pipe: No sensor fusion - pipe GPS for position and IMU for bearing
* Q_dynamic: LKF process noise when moving
* Q_static: LKF process noise when still
* P_initial: LKF initial state estimate covariance
* R_dynamic: LKF sensor noise when moving
* R_static: LKF sensor noise when still
* dt: Time step in seconds
* UpdateRate: Update rate in seconds per update
* IMU_fresh_timeout: Seconds after which an IMU measurement is considered stale
* GPS_fresh_timeout: Seconds after which an GPS measurement is considered stale
* IMU_accel_filter_bias: IMU accelerometer low-pass filter bias
* IMU_accel_threshold: IMU accelerometer threshold-to-zero threshold
* RefCoords: GPS reference coordinates for conversion to and from meters

### Code Overview

**`__main__.py`**: Handles LCM interfaces and calls sensor fusion code
**`__init__.py`**: Empty file to create Python module
**`conversions.py`**: Useful unit conversion functions
**`inputs.py`**: Objects for each sensor
**`lkf.py`**: Implementation of a Linear Kalman Filter