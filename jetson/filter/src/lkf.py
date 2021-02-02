import numpy as np
from copy import deepcopy
from scipy.linalg import block_diag
from .conversions import *


class LKFFusion:
    '''
    Class for GPS and IMU fusion using an LKF

    @attribute dict config: LKF parameters
    '''

    def __init__(self, state_estimate, config, ref_lat=0, ref_long=0):
        '''
        Constructs the filter

        @param StateEstimate state_estimate: initial state estimate
        @param dict config: LKF parameters
        @param ref_lat: reference latitude used to reduce conversion error
        @param ref_long: reference longitude used to reduce conversion error
        '''
        self.config = config
        dt = self.config["dt"]

        self.ref_lat = ref_lat
        self.ref_long = ref_long

        F = np.array([[1., dt, 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., dt],
                      [0., 0., 0., 1.]])

        B = np.array([[0.5*dt**2., 0.],
                      [dt, 0.],
                      [0., 0.5*dt**2.],
                      [0., dt]])

        H = np.eye(4)

        Q = QDiscreteWhiteNoise(2, dt, self.config["Q_Dynamic"], 2)
        P_initial = self.config['P_initial']
        R = self.config['R_Dynamic']

        pos_meters = state_estimate.posToMeters()
        abs_vel = state_estimate.separateSpeed()
        x_initial = np.array([pos_meters["lat"], abs_vel["north"], pos_meters["long"], abs_vel["east"]])

        self.lkf = LinearKalmanFilter(4, 4, x_initial, P_initial, F, H, Q, R, dim_u=2, B=B)

    def iterate(self, state_estimate, pos, vel, accel, bearing, pitch_deg, static=False, use_vel=None):
        '''
        Runs one iteration of the LKF

        @param StateEstimate state_estimate: state estimate to update
        @param PosComponent pos: fresh position estimate
        @param VelComponent vel: fresh speed estimate
        @param AccelComponent accel: fresh acceleration estimate
        @param BearingComponent bearing: fresh bearing estimate
        @param float pitch_deg: fresh pitch estimate
        @param bool static: current nav state is static
        @param bool/None use_vel: use velocity in update step
        '''
        if use_vel is None:
            use_vel = self.config["use_vel"]

        u = None
        if accel is not None:
            abs_accel = accel.absolutify()
            u = np.array([abs_accel["north"], abs_accel["east"]])
        Q = QDiscreteWhiteNoise(2, self.config["dt"], self.config["Q_Static"], 2) if not static else None
        self.lkf.predict(u=u, Q=Q)

        if static:
            self._zeroVel()

        z = None
        H = None
        vel = vel if use_vel else None
        if pos is not None:
            pos_meters = {}
            pos_decimal = pos.asDegs()
            pos_meters["long"] = long2meters(pos_decimal["long"], pos_decimal["lat"],
                                             ref_long=self.ref_long)
            pos_meters["lat"] = lat2meters(pos_decimal["lat"],
                                           ref_lat=self.ref_lat)
            if vel is not None:
                # If both position and velocity are available, use both
                abs_vel = vel.absolutify(bearing.bearing_deg, None)
                z = np.array([pos_meters["lat"], abs_vel["north"], pos_meters["long"], abs_vel["east"]])
                H = np.eye(4)
            else:
                # If only position is available, zero out the velocity residual
                abs_vel = state_estimate.separateSpeed()
                z = np.array([pos_meters["lat"], abs_vel["north"], pos_meters["long"], abs_vel["east"]])
                H = np.diag([1, 0, 1, 0])
        else:
            if vel is not None:
                # If only velocity is availble, zero out the position residual
                pos_meters = state_estimate.posToMeters()
                abs_vel = vel.absolutify(bearing.bearing_deg, None)
                z = np.array([pos_meters["lat"], abs_vel["north"], pos_meters["long"], abs_vel["east"]])
                H = np.diag([0, 1, 0, 1])

        R = None
        if static:
            R = self.config["R_Static"]
        if z is not None:
            self.lkf.update(z, H=H, R=R)

        if static:
            self._zeroVel()

        print(self.lkf.x)

        speed = np.hypot(self.lkf.x[1], self.lkf.x[3])
        state_estimate.updateFromMeters(self.lkf.x[0], self.lkf.x[2], speed, bearing.bearing_deg)

    def _zeroVel(self):
        '''
        Zeros velocity and variance in velocity
        '''
        self.lkf.x[1] = 0.0
        self.lkf.x[3] = 0.0
        self.lkf.P[1,:] = 0.0
        self.lkf.P[3,:] = 0.0

class LinearKalmanFilter:
    '''
    Class for Linear Kalman filtering

    @attribute int dim_x: dimension of state space
    @attribute int dim_z: dimension of measurement space
    @attribute int dim_u: dimension of control input
    @attribute ndarray x: state vector
    @attribute ndarray P: state covariance matrix
    @attribute ndarray z: measurement vector
    @attribute ndarray R: measurement noise matrix
    @attribute ndarray Q: process noise matrix
    @attribute ndarray F: state transition matrix
    @attribute ndarray B: control transition matrix
    @attribute ndarray H: state space -> measurement space transform matrix
    @attribute ndarray _I: identity matrix
    @attribute function inv: matrix inverse function
    '''

    def __init__(self, dim_x, dim_z, x_initial, P_initial, F, H, Q, R, dim_u=0, B=None, p_inverse=True):
        '''
        Initializes filter matrices

        @param int dim_x: dimension of state space
        @param int dim_z: dimension of measurement space
        @param ndarray x_initial: initial state estimate
        @param list P_initial: initial state estimate covariance
        @param ndarray F: state transition matrix
        @param ndarray H: state space -> measurement space transform matrix
        @param ndarray Q: process noise matrix
        @param list R: measurement noise for each measurement variable
        @param int dim_u: dimension of control input
        @param ndarray B: control transition matrix
        @param bool p_inverse: use Moore-Penrose pseudo-inverse
        '''
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        x_initial = np.atleast_2d(x_initial)
        if x_initial.shape == (1, self.dim_x):
            x_initial = x_initial.T
        if x_initial.shape != (self.dim_x, 1):
            raise ValueError("x_initial dimension mismatch")
        self.x = x_initial

        if len(P_initial) != self.dim_x:
            raise ValueError("P_initial dimension mismatch")
        self.P = np.diag(P_initial)

        self.F = deepcopy(F)
        if self.F.shape != (self.dim_x, self.dim_x):
            raise ValueError("F dimension mismatch")

        if B is not None:
            self.B = deepcopy(B)
            if self.B.shape != (self.dim_x, self.dim_u):
                raise ValueError("B dimension mismatch")
        else:
            self.B = None

        self.H = deepcopy(H)
        if self.H.shape != (self.dim_z, self.dim_x):
            raise ValueError("H dimension mismatch")

        self.Q = deepcopy(Q)
        if self.Q.shape != (self.dim_x, self.dim_x):
            raise ValueError("Q dimension mismatch")

        if len(R) != self.dim_z:
            raise ValueError("R dimension mismatch")
        self.R = np.diag(R)

        self._I = np.eye(dim_x)

        self.inv = np.linalg.pinv if p_inverse else np.linalg.inv

    def predict(self, u=None, B=None, F=None, Q=None):
        '''
        Predict forward to get priors

        @param ndarray u: control input vector
        @param ndarray B: control transition matrix
        @param ndarray F: state transition matrix
        @param float/ndarray Q: process noise matrix
        '''

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = self._I * Q

        # State prediction: x = Fx + Bu
        if u is not None:
            u = np.atleast_2d(u)
            if u.shape == (1, self.dim_u):
                u = u.T
            if u.shape != (self.dim_u, 1):
                raise ValueError("u dimension mismatch")
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # Covariance prediction: P = FPF' + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, z, R=None, H=None):
        '''
        Update the filter with a measurement

        @param ndarray z: measurement vector
        @param float/list R: measurement noise for each measurement variable
        @param ndarray H: state space -> measurement space transform matrix
        '''
        z = np.atleast_2d(z)
        if z.shape == (1, self.dim_z):
            z = z.T
        if z.shape != (self.dim_z, 1):
            raise ValueError("z dimension mismatch")

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R
        else:
            R = np.diag(R)

        if H is None:
            H = self.H

        # Residual: y = z - Hx
        y = z - np.dot(H, self.x)

        PHT = np.dot(self.P, H.T)
        # System uncertainty in measurement space: S = HPH' + R
        S = np.dot(H, PHT) + R
        # Kalman gain: K = PH'/S
        S_inverse = self.inv(S)
        K = np.dot(PHT, S_inverse)

        # State estimate: x = x + Ky
        self.x = self.x + np.dot(K, y)

        # Covariance estimate: P = (I-KH)P(I-KH)' + KRK'
        # P = (I-KH)P usually seen in the literature
        # "This is more numerically stable and works for non-optimal K vs the equation" - filterpy
        I_KH = self._I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)


def QDiscreteWhiteNoise(dim, dt, variance=1., block_size=1):
    '''
    Returns the Q matrix for the Discrete Constant White Noise Model
    Q is computed as G * G.T * variance, where G is the process noise per time step

    Copied from filterpy

    @param int dim: dimensions in Q
    @param float dt: timestep
    @param float variance: variance in noise
    @param int block_size: number of derivatives in one dimension
    @return ndarray: white noise matrix
    '''

    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [.5*dt**3, dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [.5*dt**3, dt**2, dt],
             [.5*dt**2, dt, 1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4, (dt**3)/2, (dt**2)/2],
             [(dt**4)/6, (dt**3)/2, dt**2, dt],
             [(dt**3)/6, (dt**2)/2, dt, 1.]]

    return block_diag(*[Q]*block_size) * variance