import numpy as np


class ConstVelocityKalmanFilter:
    """
    Kalman Filter for tracking velocity states.

    State vector: [vx, vy, vz, wx, wy, wz]
    Measurement vector: same as state (direct velocity measurements)
    """

    def __init__(
        self,
        x_init,
        # Assume car mena acceleration is 1 m/s2
        # Velocity can change by 0.1 m/s per 0.1 sec
        # covariance is 0.1^2 = 0.01
        linear_process_noise_covariance=0.01,
        # Alpha Prime Lidar has 3 cm accuracy
        # https://visimind.com/wp-content/uploads/pdf/table_with_sensors.pdf
        # 0.03 m is divided by 0.1 sec, linear velocity accuracy is 0.3 m/s
        # covariance is 0.3^2 = 0.09
        angular_process_noise_covariance=0.09,
        linear_measurement_noise_covariance=0.01,
        # Let's take point at 100 meter range, and accuracy 3 cm
        # Then angle is 0.03 / 100 = 3e-4 radians
        # 3e-4 is divided by 0.1 sec, angular velocity accuracy is 3e-3 m/s
        # covariance is 3e-3^2 = 9e-6
        angular_measurement_noise_covariance=9e-6,
    ):
        """
        Initialize the Kalman filter.

        Args:
            process_noise_covariance: Q - Process noise covariance
            measurement_noise_covariance: R - Measurement noise covariance
        """
        self.dim = 6  # Number of state variables (vx, vy, vz, wx, wy, wz)

        # Process noise covariance matrix Q
        self.Q = np.eye(self.dim)
        self.Q[0:3, 0:3] *= linear_process_noise_covariance
        self.Q[3:6, 3:6] *= angular_process_noise_covariance

        # Measurement noise covariance matrix R
        self.R = np.eye(self.dim)
        self.R[0:3, 0:3] *= linear_measurement_noise_covariance
        self.R[3:6, 3:6] *= angular_measurement_noise_covariance

        # State transition matrix F (identity for constant velocity model)
        self.F = np.eye(self.dim)

        # Measurement matrix H (identity - we measure all states directly)
        self.H = np.eye(self.dim)

        # Initial state estimate
        self.x = x_init

        # Initial error covariance matrix P
        # We assume, that initial velocities are GT
        self.P = np.zeros((self.dim, self.dim))

    def predict(self):
        """
        Predict step: x_pred = F * x, P_pred = F * P * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update step: K = P * H^T * (H * P * H^T + R)^-1
                   x = x_pred + K * (z - H * x_pred)
                   P = (I - K * H) * P

        Args:
            z: Measurement vector
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        self.K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + self.K @ y

        # Update error covariance
        I_KH = np.eye(self.dim) - self.K @ self.H
        self.P = I_KH @ self.P
