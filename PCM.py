
import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
A = 1.0
f = 5                 # Signal frequency (Hz)
fs = 100               # Sampling frequency (Hz)
n_bits = 3             # Bits per sample
L = 2 ** n_bits        # Quantization levels
duration = 1.0

# 2. Generate original signal
t = np.linspace(0, duration, 2000, endpoint=False)
x = A * np.sin(2 * np.pi * f * t)

# 3. Sampling
ts = np.arange(0, duration, 1 / fs)
samples = A * np.sin(2 * np.pi * f * ts)

# 4. Quantization
v_min = -A
v_max = A
delta = (v_max - v_min) / L
print(f"delta : {delta}")
indices = np.floor((samples - v_min) / delta).astype(int)
indices = np.clip(indices, 0, L - 1)

quantized = v_min + (indices + 0.5) * delta

# 5. PCM Encoding: convert each level to binary
pcm_words = [format(i, f"0{n_bits}b") for i in indices]
bitstream = "".join(pcm_words)

# 6. PCM Decoding / Demodulation
decoded_indices = np.array(
    [int(word, 2) for word in pcm_words]
)

reconstructed = (
    v_min + (decoded_indices + 0.5) * delta
)

# 7. Performance measurements
error = samples - reconstructed
mse = np.mean(error ** 2)
bit_rate = fs * n_bits

print("PULSE CODE MODULATION RESULTS")
print("-" * 35)
print("Sampling frequency:", fs, "Hz")
print("Bits per sample:", n_bits)
print("Quantization levels:", L)
print("Quantization step:", delta, "V")
print("PCM bit rate:", bit_rate, "bits/second")
print("Number of samples:", len(samples))
print("Mean squared error:", round(mse, 5))
print("First 60 PCM bits:", bitstream[:60])

# 8. Plot all stages
fig, ax = plt.subplots(4, 1, figsize=(12, 12))

ax[0].plot(t, x, label="Original signal")
ax[0].stem(ts, samples, linefmt="C1-",
           markerfmt="C1o", basefmt=" ")
ax[0].set_title("Original and Sampled Signal")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t, x, "--", label="Original")
ax[1].step(ts, quantized, where="mid",
           label="Quantized")
ax[1].set_title("Quantization")
ax[1].set_ylabel("Amplitude")
ax[1].legend()
ax[1].grid(True)

ax[2].plot(ts, samples, "o-", label="Sampled")
ax[2].step(ts, reconstructed, where="mid",
           label="Demodulated")
ax[2].set_title("PCM Demodulation and Reconstruction")
ax[2].set_ylabel("Amplitude")
ax[2].legend()
ax[2].grid(True)

ax[3].plot(ts, error, "o-", markersize=3)
ax[3].set_title(f"Quantization Error (MSE = {mse:.5f})")
ax[3].set_xlabel("Time (seconds)")
ax[3].set_ylabel("Error")
ax[3].grid(True)

plt.tight_layout()
plt.show()
