
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# 1. Parameters
num_bits = 8
chips_per_bit = 8
noise_std = 0.35

# 2. Generate random binary data
data_bits = np.random.randint(0, 2, num_bits)

# Convert: 0 -> -1, 1 -> +1
data_symbols = 2 * data_bits - 1

# 3. Generate pseudorandom spreading code
spreading_code = np.random.choice(
    [-1, 1], chips_per_bit
)

# 4. Spread the data
spread_signal = (
    np.repeat(data_symbols, chips_per_bit)
    * np.tile(spreading_code, num_bits)
)

# 5. Transmit through a noisy channel
noise = np.random.normal(
    0, noise_std, len(spread_signal)
)
received_signal = spread_signal + noise

# 6. Despread the received signal
received_blocks = received_signal.reshape(
    num_bits, chips_per_bit
)

despread_values = np.mean(
    received_blocks * spreading_code, axis=1
)

# 7. Recover original binary data
recovered_bits = (despread_values > 0).astype(int)

# 8. Display results
print("DSSS SIMULATION RESULTS")
print("-" * 40)
print("Original data: ", data_bits)
print("Spreading code:", spreading_code)
print("Chips per bit:", chips_per_bit)
print("Total chips:", len(spread_signal))
print("Despread values:", np.round(despread_values, 3))
print("Recovered data:", recovered_bits)
print(
    "Successful recovery:",
    np.array_equal(data_bits, recovered_bits)
)

# 9. Plot the signals
fig, ax = plt.subplots(4, 1, figsize=(12, 12))

# Original data
ax[0].step(
    np.arange(num_bits + 1),
    np.r_[data_symbols, data_symbols[-1]],
    where="post"
)
ax[0].set_title("1. Original Binary Data")
ax[0].set_ylabel("Amplitude")
ax[0].set_yticks([-1, 1], labels=["0 (-1)", "1 (+1)"])
ax[0].grid(True)

# Spread signal
chip_time = np.arange(len(spread_signal))
ax[1].step(chip_time, spread_signal, where="post")
ax[1].set_title("2. DSSS Spread Signal")
ax[1].set_ylabel("Amplitude")
ax[1].grid(True)

# Received signal with noise
ax[2].plot(chip_time, received_signal, ".-")
ax[2].axhline(0, linestyle="--")
ax[2].set_title("3. Received Signal with Noise")
ax[2].set_ylabel("Amplitude")
ax[2].grid(True)

# Despread decisions
ax[3].stem(np.arange(num_bits), despread_values)
ax[3].axhline(0, linestyle="--")
ax[3].set_title("4. Despread Signal")
ax[3].set_xlabel("Data bit index")
ax[3].set_ylabel("Decision value")
ax[3].grid(True)

plt.tight_layout()
plt.show()
