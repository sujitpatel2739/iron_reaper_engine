import numpy as np
import matplotlib.pyplot as plt

def analyze_trend(data, threshold=0.001):
    """
    Analyzes the trend of a list of numbers using Linear Regression.
    
    Args:
        data (list or np.array): The list of 1000 numbers.
        threshold (float): The slope value below which we consider the trend 'Constant'.
                           Adjust this based on the scale of your data.
    """
    # 1. Create the X-axis (Time steps: 0, 1, 2, ... N)
    x = np.arange(len(data))
    y = np.array(data)

    # 2. Linear Regression: Fit a line (y = mx + c)
    # np.polyfit(x, y, 1) returns [slope (m), intercept (c)]
    slope, intercept = np.polyfit(x, y, 1)

    # 3. Determine the Trend based on Slope
    if slope > threshold:
        prediction = "INCREASING 📈"
    elif slope < -threshold:
        prediction = "DECREASING 📉"
    else:
        prediction = "CONSTANT (AVERAGE) ➡️"

    # 4. Print Statistics
    print(f"--- Trend Analysis ---")
    print(f"Calculated Slope (m): {slope:.5f}")
    print(f"Prediction: {prediction}")
    
    # 5. Optional: Compare First Half vs Second Half (The Simple Method)
    mid = len(y) // 2
    mean_first = np.mean(y[:mid])
    mean_last = np.mean(y[mid:])
    print(f"Mean (First 50%): {mean_first:.2f}")
    print(f"Mean (Last 50%):  {mean_last:.2f}")

    return x, y, slope, intercept, prediction

# ==========================================
# Test with Dummy Data
# ==========================================

# 1. Generate Random Data (e.g., Noisy Increasing)
# np.random.seed(42)
steps = 1000
noise = np.random.normal(0, 10, steps)  # Random noise
trend = np.random.randn(steps).cumsum() * 0.1
# Actual trend (0 to 20)
data = trend + noise                    # Combine them

# 2. Run Analysis
x, y, m, c, pred = analyze_trend(data)

# 3. Visualize (Graph)
plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=2, color='gray', alpha=0.5, label='Noisy Data') # The dots
plt.plot(x, m*x + c, color='red', linewidth=3, label=f'Trend Line (Slope={m:.4f})') # The Line
plt.title(f"Detected Trend: {pred}")
plt.legend()
plt.show()