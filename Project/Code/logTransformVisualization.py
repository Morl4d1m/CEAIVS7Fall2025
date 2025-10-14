import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
i = 0
output = []

# Loop over 8-bit values (0–255)
while i < 256:
    # Compute scaling constant dynamically for demonstration
    c = 255 / np.log(1 + 255)   # Usually constant (based on max value)
    
    # Compute log transform
    log_transformed = c * np.log(1 + i)
    
    # Append result to list
    output.append(log_transformed)
    
    # Increment
    i += 1

# Convert to NumPy array for plotting
output = np.array(output)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(256), output, color='blue', linewidth=2)
plt.title("Log Transformation", fontsize=14)
plt.xlabel("Input intensity (0–255)")
plt.ylabel("Output intensity (log-transformed)")
plt.grid(True)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.show()
