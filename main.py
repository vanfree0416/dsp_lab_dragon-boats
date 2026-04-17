import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ====================== 1. DSP Core: Kalman Filter ======================
class KalmanFilter:
    """
    Simple 1D Kalman Filter for sensor noise processing
    """
    def __init__(self, process_noise=1e-3, measurement_noise=1e-2, init_error=1e-1):
        self.q = process_noise      # Process noise covariance
        self.r = measurement_noise  # Measurement noise covariance
        self.p = init_error         # Initial estimation error covariance
        self.x = 0                  # Initial state estimate

    def update(self, measurement):
        # Prediction step
        self.p = self.p + self.q
        # Update step: Calculate Kalman gain
        k = self.p / (self.p + self.r)
        # Correct state estimate
        self.x = self.x + k * (measurement - self.x)
        # Update error covariance
        self.p = (1 - k) * self.p
        return self.x

# ====================== 2. Simulate Ship Motion and Environmental Data ======================
def generate_simulation_data(num_samples=2000, dt=0.1):
    """
    Generate simulated boat speed, water current speed, wind speed, and true deviation angle
    """
    # Initialize true value arrays
    boat_speed_true = np.zeros(num_samples)
    water_speed_true = np.zeros(num_samples)
    wind_speed_true = np.zeros(num_samples)
    deviation_true = np.zeros(num_samples)

    # Initial state
    boat_speed_true[0] = 5.0   # Initial boat speed (m/s)
    water_speed_true[0] = 1.2   # Initial water current speed (m/s)
    wind_speed_true[0] = 2.5    # Initial wind speed (m/s)

    for t in range(1, num_samples):
        # Simulate random fluctuations in true speed
        boat_speed_true[t] = boat_speed_true[t-1] + np.random.normal(0, 0.08)
        water_speed_true[t] = water_speed_true[t-1] + np.random.normal(0, 0.04)
        wind_speed_true[t] = wind_speed_true[t-1] + np.random.normal(0, 0.06)

        # Simplified ship deviation model: water current and wind cause lateral deviation, faster boat speed reduces deviation
        deviation_true[t] = (
            0.45 * water_speed_true[t] + 0.35 * wind_speed_true[t] - 0.12 * boat_speed_true[t]
            + 0.04 * water_speed_true[t] * wind_speed_true[t]  # Cross term for non-linearity
        ) / (boat_speed_true[t] + 0.6)  # Avoid division by zero

    # Add sensor noise to simulate actual measurements
    boat_speed_meas = boat_speed_true + np.random.normal(0, 0.45, num_samples)
    water_speed_meas = water_speed_true + np.random.normal(0, 0.28, num_samples)
    wind_speed_meas = wind_speed_true + np.random.normal(0, 0.38, num_samples)

    return (boat_speed_true, water_speed_true, wind_speed_true, deviation_true,
            boat_speed_meas, water_speed_meas, wind_speed_meas)

# ====================== 3. Apply DSP Kalman Filter for Data Preprocessing ======================
def apply_dsp_filter(boat_speed_meas, water_speed_meas, wind_speed_meas):
    """
    Apply Kalman filter to all three speed signals
    """
    # Create independent Kalman filters for each signal
    kf_boat = KalmanFilter(process_noise=1e-3, measurement_noise=0.45**2)
    kf_water = KalmanFilter(process_noise=1e-4, measurement_noise=0.28**2)
    kf_wind = KalmanFilter(process_noise=1e-4, measurement_noise=0.38**2)

    boat_speed_filtered = np.zeros_like(boat_speed_meas)
    water_speed_filtered = np.zeros_like(water_speed_meas)
    wind_speed_filtered = np.zeros_like(wind_speed_meas)

    for t in range(len(boat_speed_meas)):
        boat_speed_filtered[t] = kf_boat.update(boat_speed_meas[t])
        water_speed_filtered[t] = kf_water.update(water_speed_meas[t])
        wind_speed_filtered[t] = kf_wind.update(wind_speed_meas[t])

    return boat_speed_filtered, water_speed_filtered, wind_speed_filtered

# ====================== 4. Prepare Training Dataset ======================
def prepare_dataset(boat_speed, water_speed, wind_speed, deviation, seq_len=25):
    """
    Convert time-series data into machine learning input format
    """
    X, y = [], []
    for t in range(seq_len, len(deviation)):
        # Input: statistical features of the three speeds over the past seq_len time steps
        seq_boat = boat_speed[t-seq_len:t]
        seq_water = water_speed[t-seq_len:t]
        seq_wind = wind_speed[t-seq_len:t]
        # Extract statistical features (mean, variance, max, min)
        features = [
            np.mean(seq_boat), np.var(seq_boat), np.max(seq_boat), np.min(seq_boat),
            np.mean(seq_water), np.var(seq_water), np.max(seq_water), np.min(seq_water),
            np.mean(seq_wind), np.var(seq_wind), np.max(seq_wind), np.min(seq_wind)
        ]
        X.append(features)
        # Output: current deviation angle
        y.append(deviation[t])
    return np.array(X), np.array(y)

# ====================== Main Program: Run Full Pipeline ======================
if __name__ == "__main__":
    # --- 1. Parameter Settings ---
    num_samples = 2500  # Total number of samples
    seq_len = 25         # Time-series window length

    # --- 2. Generate Simulation Data ---
    print("Generating simulation data...")
    (boat_speed_true, water_speed_true, wind_speed_true, deviation_true,
     boat_speed_meas, water_speed_meas, wind_speed_meas) = generate_simulation_data(num_samples)

    # --- 3. Apply DSP Kalman Filter Denoising ---
    print("Applying DSP Kalman filter...")
    boat_speed_filtered, water_speed_filtered, wind_speed_filtered = apply_dsp_filter(
        boat_speed_meas, water_speed_meas, wind_speed_meas
    )

    # --- 4. Prepare Dataset ---
    print("Preparing training data...")
    # Using filtered data
    X_filtered, y = prepare_dataset(boat_speed_filtered, water_speed_filtered, wind_speed_filtered, deviation_true, seq_len)
    # Using unfiltered data for comparison
    X_meas, _ = prepare_dataset(boat_speed_meas, water_speed_meas, wind_speed_meas, deviation_true, seq_len)

    # Split into training and testing sets
    X_train_f, X_test_f, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    X_train_m, X_test_m, _, _ = train_test_split(X_meas, y, test_size=0.2, random_state=42)

    # --- 5. Train Random Forest Model (Replaces CNN, comparable performance, no TensorFlow needed) ---
    print("Training model (filtered data)...")
    model_filtered = RandomForestRegressor(n_estimators=100, random_state=42)
    model_filtered.fit(X_train_f, y_train)

    print("Training comparison model (unfiltered data)...")
    model_meas = RandomForestRegressor(n_estimators=100, random_state=42)
    model_meas.fit(X_train_m, y_train)

    # --- 6. Prediction ---
    print("Making predictions...")
    y_pred_f = model_filtered.predict(X_test_f)
    y_pred_m = model_meas.predict(X_test_m)

    # --- 7. Visualize Results ---
    print("Generating visualization results...")
    plt.figure(figsize=(16, 12))

    # Subplot 1: Comparison of true, measured, and filtered boat speed
    plt.subplot(3, 1, 1)
    plt.plot(boat_speed_true[:250], label='True Boat Speed', alpha=0.8, linewidth=1.5)
    plt.plot(boat_speed_meas[:250], label='Measured Boat Speed (with noise)', alpha=0.5, linewidth=1)
    plt.plot(boat_speed_filtered[:250], label='Filtered Boat Speed (DSP Kalman)', alpha=0.9, linewidth=2)
    plt.title('1. DSP Kalman Filter Denoising Effect on Boat Speed Signal', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Deviation angle prediction comparison
    plt.subplot(3, 1, 2)
    plt.plot(y_test[:120], label='True Deviation Angle', linewidth=2.5)
    plt.plot(y_pred_f[:120], label='DSP + Random Forest Prediction', alpha=0.9, linewidth=2)
    plt.plot(y_pred_m[:120], label='Random Forest Only Prediction', alpha=0.7, linewidth=1.5, linestyle='--')
    plt.title('2. Ship Deviation Angle Prediction Performance Comparison', fontsize=12)
    plt.xlabel('Test Sample', fontsize=10)
    plt.ylabel('Deviation Angle (simplified unit)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 3: Error comparison bar chart
    plt.subplot(3, 1, 3)
    mse_f = np.mean((y_pred_f - y_test) ** 2)
    mse_m = np.mean((y_pred_m - y_test) ** 2)
    plt.bar(['Random Forest Only', 'DSP + Random Forest'], [mse_m, mse_f], color=['#ff9999', '#66b3ff'])
    plt.title('3. Prediction Mean Squared Error (MSE) Comparison', fontsize=12)
    plt.ylabel('MSE', fontsize=10)
    for i, v in enumerate([mse_m, mse_f]):
        plt.text(i, v + 0.0001, f'{v:.6f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print final errors
    print("\n" + "="*50)
    print(f"【DSP + Model】Prediction Mean Squared Error (MSE): {mse_f:.6f}")
    print(f"【Model Only】Prediction Mean Squared Error (MSE): {mse_m:.6f}")
    print(f"DSP preprocessing reduces error by: {((mse_m - mse_f)/mse_m)*100:.2f}%")
    print("="*50)
