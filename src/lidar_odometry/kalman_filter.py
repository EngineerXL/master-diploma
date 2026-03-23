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
        linear_measurement_noise_covariance=0.09,
        # idk, just trust angular velocities
        angular_process_noise_covariance=0.01,
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
        self.process_cov_Q = np.eye(self.dim)
        self.process_cov_Q[0:3, 0:3] *= linear_process_noise_covariance
        self.process_cov_Q[3:6, 3:6] *= angular_process_noise_covariance

        # Measurement noise covariance matrix R
        self.observation_cov_R = np.eye(self.dim)
        self.observation_cov_R[0:3, 0:3] *= linear_measurement_noise_covariance
        self.observation_cov_R[3:6, 3:6] *= angular_measurement_noise_covariance

        # State transition matrix F (identity for constant velocity model)
        self.process_model_F = np.eye(self.dim)

        # Measurement matrix H (identity - we measure all states directly)
        self.observation_model_H = np.eye(self.dim)

        # Initial state estimate
        self.estimate_x = x_init

        # Initial error covariance matrix P
        # We assume, that initial velocities are GT
        self.state_cov_P = np.zeros((self.dim, self.dim))

    def predict(self):
        """
        Predict step: x_pred = F * x, P_pred = F * P * F^T + Q
        """
        self.estimate_x = self.process_model_F @ self.estimate_x
        # cov(x) = P
        # cov(F * x) = F cov(x) * F^T = F * P * F^T
        self.state_cov_P = (
            self.process_model_F @ self.state_cov_P @ self.process_model_F.T
            + self.process_cov_Q
        )

    def get_estimate(self):
        return self.estimate_x

    def update(self, z):
        """
        Update step: K = P * H^T * (H * P * H^T + R)^-1
                   x = x_pred + K * (z - H * x_pred)
                   P = (I - K * H) * P

        Args:
            z: Measurement vector
        """
        # Innovation (measurement residual)
        y = z - self.observation_model_H @ self.estimate_x

        # Kalman gain
        up = self.state_cov_P @ self.observation_model_H.T
        down = (
            self.observation_model_H @ self.state_cov_P @ self.observation_model_H.T
            + self.observation_cov_R
        )
        # If observations are acurate (R is small), then we trust observations (K -> 1)
        # On the other side (R is large) we trust our model prediction (K -> 0)
        self.gain_K = up @ np.linalg.inv(down)

        # Update state estimate
        self.estimate_x = self.estimate_x + self.gain_K @ y

        # Update error covariance
        I_KH = np.eye(self.dim) - self.gain_K @ self.observation_model_H
        self.state_cov_P = I_KH @ self.state_cov_P
