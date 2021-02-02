import json
import time
import asyncio
import numpy as np
from copy import deepcopy
from os import getenv
from rover_common import aiolcm
from rover_common.aiohelper import run_coroutines
from rover_msgs import IMUData, GPS, Odometry, NavStatus
from .inputs import Gps, Imu
from .lkf import LKFFusion
from .conversions import *


class StateEstimate:
    '''
    Class for current state estimate

    @attribute dict pos: current position estimate (integer degrees, decimal minutes)
    @attribute float speed: current ground speed estimate (m/s)
    @attribute float bearing_deg: current bearing estimate (decimal degrees East of North)
    @attribute float ref_lat: reference latitude used to reduce conversion error
    @attribute float ref_long: reference longitude used to reduce conversion error
    '''

    def __init__(self, lat_deg, lat_min, long_deg, long_min, speed, bearing_deg,
                 ref_lat=0, ref_long=0):
        '''
        Initalizes state variable values

        @param int lat_deg: latitude integer degrees
        @param float lat_min: latitude decimal minutes
        @param int long_deg: longitude integer degrees
        @param float long_min: longitude decimal minutes
        @param float speed: ground speed (m/s)
        @param float bearing_deg: absolute bearing (decimal degrees East of North)
        @param float ref_lat: reference latitude used to reduce conversion error
        @param float ref_long: reference longitude used to reduce conversion error
        '''
        self.pos = {"lat_deg": lat_deg, "lat_min": lat_min, "long_deg": long_deg,
                    "long_min": long_min}
        self.speed = speed
        self.bearing_deg = bearing_deg
        self.ref_lat = ref_lat
        self.ref_long = ref_long

    def posToMeters(self):
        '''
        Returns the current position estimate converted to meters

        @return dict: current position estimate (meters)
        '''
        pos_meters = {}
        pos_meters["long"] = min2decimal(self.pos["long_deg"], self.pos["long_min"])
        pos_meters["lat"] = min2decimal(self.pos["lat_deg"], self.pos["lat_min"])
        pos_meters["long"] = long2meters(pos_meters["long"], pos_meters["lat"],
                                         ref_long=self.ref_long)
        pos_meters["lat"] = lat2meters(pos_meters["lat"], ref_lat=self.ref_lat)
        return pos_meters

    def separateSpeed(self):
        '''
        Returns the ground speed separated into absolute components

        @return dict: current absolute velocity estimate (m/s)
        '''
        abs_vel = {}
        abs_vel["north"] = self.speed * np.cos(np.radians(bearing_deg))
        abs_vel["east"] = self.speed * np.sin(np.radians(bearing_deg))
        return abs_vel

    def updateFromMeters(self, lat, long, speed, bearing_deg):
        '''
        Updates state estimate from the filter output

        @param float lat: new latitude estimate (meters)
        @param float long: new longitude estimate (meters)
        @param float speed: new ground speed estimate (m/s)
        @param float bearing_deg: new bearing estimate (degrees East of North)
        '''
        lat_decimal_deg = meters2lat(lat, ref_lat=self.ref_lat)
        self.pos["lat_deg"], self.pos["lat_min"] = decimal2min(lat_decimal_deg)
        long_decimal_deg = meters2long(long, lat_decimal_deg, ref_long=self.ref_long)
        self.pos["long_deg"], self.pos["long_min"] = decimal2min(long_decimal_deg)
        self.speed = speed
        self.bearing_deg = bearing_deg

    def asOdom(self):
        '''
        Returns the current state estimate as an Odometry LCM object

        @return Odometry: state estimate in Odometry LCM format
        '''
        odom = Odometry()
        odom.latitude_deg = self.pos["lat_deg"]
        odom.latitude_min = self.pos["lat_min"]
        odom.longitude_deg = self.pos["long_deg"]
        odom.longitude_min = self.pos["long_min"]
        odom.bearing_deg = self.bearing_deg
        odom.speed = np.hypot(self.vel["north"], self.vel["east"])
        return odom


class SensorFusion:
    '''
    Class for filtering sensor data and outputting state estimates

    @attribute dict config: user-configured parameters found in config/filter/config.json
    @attribute Gps gps: GPS sensor
    @attribute Imu imu: IMU sensor
    @attribute str nav_state: current nav state
    @attribute set static_nav_states: nav states where rover is known to be static
    @attribute Filter filter: filter used to perform sensor fusion
    @attribute StateEstimate state_estimate: current state estimate
    @attribute AsyncLCM lcm: LCM interface
    '''

    def __init__(self):
        config_path = getenv('MROVER_CONFIG')
        config_path += "/config_filter/config.json"
        with open(config_path, "r") as config:
            self.config = json.load(config)

        self.gps = Gps()
        self.imu = Imu(self.config["IMU_accel_filter_bias"], self.config["IMU_accel_threshold"])
        self.nav_state = None
        self.static_nav_states = {None, "Off", "Done", "Search Spin Wait", "Turned to Target Wait", "Gate Spin Wait",
                                  "Turn", "Search Turn", "Turn to Target", "Turn Around Obstacle", "Search Turn Around Obstacle",
                                  "Gate Turn", "Gate Turn to Center Point", "Radio Repeater Turn"}

        self.filter = None
        self.state_estimate = None

        self.lcm = aiolcm.AsyncLCM()
        self.lcm.subscribe("/gps", self._gpsCallback)
        self.lcm.subscribe("/imu_data", self._imuCallback)
        self.lcm.subscribe("/nav_status", self._navStatusCallback)

    def _gpsCallback(self, channel, msg):
        new_gps = GPS.decode(msg)
        self.gps.update(new_gps)

        # Attempt to filter on first GPS message
        if self.filter is None and self.gps.ready():
            self._constructFilter()
            self.gps.fresh = False

    def _imuCallback(self, channel, msg):
        new_imu = IMUData.decode(msg)
        self.imu.update(new_imu)

    def _navStatusCallback(self, channel, msg):
        new_nav_status = NavStatus.decode(msg)
        self.nav_state = new_nav_status.nav_state_name

    def _constructFilter(self):
        '''
        Constructs filter depending on filter type
        '''
        bearing = self._getFreshBearing()
        pitch = self._getFreshPitch()
        pos = self._getFreshPos()
        vel = self._getFreshVel()

        if bearing is None or pos is None or vel is None or (not vel.ground and pitch is None):
            return

        bearing = bearing.bearing_deg
        pos = pos.asDegsMins()
        if self.nav_state in self.static_nav_states:
            vel = 0
        else:
            vel = vel.flatten(pitch)

        self.state_estimate = StateEstimate(pos["lat_deg"], pos["lat_min"],
                                            pos["long_deg"], pos["long_min"],
                                            vel, bearing,
                                            ref_lat=self.config["RefCoords"]["lat"],
                                            ref_long=self.config["RefCoords"]["long"])

        if self.config["FilterType"] == "LKF":
            config_path = getenv('MROVER_CONFIG')
            config_path += "/config_filter/lkf_config.json"
            with open(config_path, "r") as lkf_config:
                lkf_config = json.load(lkf_config)
            lkf_config["dt"] = self.config["UpdateRate"]
            self.filter = LKFFusion(self.state_estimate, lkf_config,
                                    ref_lat=self.config["RefCoords"]["lat"],
                                    ref_long=self.config["RefCoords"]["long"])
        elif self.config["FilterType"] == "Pipe":
            self.filter = "Pipe"
        else:
            raise ValueError("Invalid filter type!")

    def _getFreshBearing(self):
        '''
        Returns a fresh bearing to use. Uses IMU over GPS, returns None if no fresh sensors

        @return BearingComponent/None: fresh bearing sensor
        '''
        for sensor in self.config["BearingPrio"]:
            if sensor == "IMU" and self.imu.fresh:
                return self.imu.bearing
            elif sensor == "GPS" and self.gps.fresh:
                return self.gps.bearing
        return None

    def _getFreshPos(self):
        '''
        Returns a fresh GPS position to use. Returns None if no fresh sensors

        @return PosComponent/None: fresh position sensor
        '''
        for sensor in self.config["PosPrio"]:
            if sensor == "GPS" and self.gps.fresh:
                return self.gps.pos
        return None

    def _getFreshVel(self):
        '''
        Returns a fresh velocity to use. Returns None if no fresh sensors

        @return VelComponent/None: fresh velocity sensor
        '''
        for sensor in self.config["VelPrio"]:
            if sensor == "GPS" and self.gps.fresh:
                return self.gps.vel
        return None

    def _getFreshAccel(self):
        '''
        Returns a fresh acceleration to use. Returns None if no fresh sensors

        @param float ref_bearing: reference bearing (decimal degrees East of North)
        @return AccelComponent/None: fresh acceleration sensor
        '''
        for sensor in self.config["AccelPrio"]:
            if sensor == "IMU" and self.imu.fresh:
                return self.imu.accel
        return None

    def _getFreshPitch(self):
        '''
        Returns a fresh pitch to use. Returns None if no fresh sensors

        @return float/None: absolute pitch (degrees)
        '''
        for sensor in self.config["PitchPrio"]:
            if sensor == "IMU" and self.imu.fresh:
                return self.imu.pitch_deg
        return None

    async def run(self):
        '''
        Runs main loop for filtering data and publishing state estimates
        '''
        while True:
            if self.filter is not None:
                # Get fresh measurements
                bearing = self._getFreshBearing()
                pitch = self._getFreshPitch()
                pos = self._getFreshPos()
                static = self.nav_state in self.static_nav_states
                vel = self._getFreshVel()
                accel = self._getFreshAccel()

                if self.config["FilterType"] == "LinearKalman":
                    self.filter.iterate(self.state_estimate, pos, vel, accel, bearing, pitch, static=static)
                elif self.config["FilterType"] == "Pipe":
                    if bearing is None or pos is None or vel is None or (not vel.ground and pitch is None):
                        return

                    bearing = bearing.bearing_deg
                    pos = pos.asDegsMins()
                    if static:
                        vel = 0
                    else:
                        vel = vel.flatten(pitch)
                    
                    self.state_estimate.pos = pos
                    self.state_estimate.speed = vel
                    self.state_estimate.bearing_deg = bearing

                odom = self.state_estimate.asOdom()
                self.lcm.publish('/odometry', odom.encode())
            else:
                self._constructFilter()
            await asyncio.sleep(self.config["UpdateRate"])


def main():
    fuser = SensorFusion()
    run_coroutines(fuser.lcm.loop(), fuser.run())


if __name__ == '__main__':
    main()