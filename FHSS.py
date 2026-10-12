
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(12)

# 1. System parameters
num_bits = 12
samples_per_bit = 200
bit_duration = 0.02
fs = samples_per_bit / bit_duration
noise_std = 0.25

# Available hopping frequencies in Hz
frequencies = np.array([200, 400, 600, 800])

# Frequency separation for binary FSK
tone_spacing = 100

# 2. Generate random binary data
data_bits = np.random.randint(0, 2, num_bits)

# 3. Generate a pseudorandom hopping sequence
hop_indices = np.random.randint(
    0, len(frequencies), num_bits
)
hop_frequencies = frequencies[hop_indices]

# 4. Generate the FHSS transmitted signal
samples_per_hop = samples_per_bit
total_samples = num_bits * samples_per_hop

t = np.arange(total_samples) / fs
transmitted = np.zeros(total_samples)

for i in range(num_bits):
    start = i * samples_per_hop
    end = start + samples_per_hop

    local_t = np.arange(samples_per_hop) / fs

    # Select the carrier for this hop
    carrier = hop_frequencies[i]

    # Binary FSK modulation:
    # Bit 0 -> carrier frequency
    # Bit 1 -> carrier frequency + tone spacing
    if data_bits[i] == 0:
        tone = carrier
    else:
        tone = carrier + tone_spacing

    transmitted[start:end] = np.cos(
        2 * np.pi * tone * local_t
    )

# 5. Simulate a noisy communication channel
noise = np.random.normal(
    0, noise_std, total_samples
)
received = transmitted + noise

# 6. Receiver: synchronize to the same hopping sequence
recovered_bits = []
decision_metrics = []

for i in range(num_bits):
    start = i * samples_per_hop
    end = start + samples_per_hop

    segment = received[start:end]
    local_t = np.arange(samples_per_hop) / fs

    carrier = hop_frequencies[i]

    # Correlate with the two possible FSK tones
    reference_0 = np.cos(
        2 * np.pi * carrier * local_t
    )
    reference_1 = np.cos(
        2 * np.pi * (carrier + tone_spacing) * local_t
    )

    corr_0 = np.sum(segment * reference_0)
    corr_1 = np.sum(segment * reference_1)

    # Select the tone with the larger correlation
    if abs(corr_1) > abs(corr_0):
        recovered_bits.append(1)
    else:
        recovered_bits.append(0)

    decision_metrics.append(
        (abs(corr_0), abs(corr_1))
    )

recovered_bits = np.array(recovered_bits)

# 7. Display results
print("=" * 50)
print("FREQUENCY HOPPING SPREAD SPECTRUM (FHSS)")
print("=" * 50)

print("Original data:    ", data_bits)
print("Hopping sequence: ", hop_frequencies, "Hz")
print("Recovered data:   ", recovered_bits)

print("Number of bits:   ", num_bits)
print("Available carriers:", frequencies, "Hz")
print("Sampling rate:    ", fs, "Hz")

errors = np.sum(data_bits != recovered_bits)
print("Bit errors:       ", errors)
print("Successful recovery:",
      np.array_equal(data_bits, recovered_bits))

# 8. Plot the results
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

bit_edges = np.arange(num_bits + 1) * bit_duration

# Plot 1: Original data
axes[0].step(
    bit_edges,
    np.r_[data_bits, data_bits[-1]],
    where="post"
)
axes[0].set_title("1. Original Binary Data")
axes[0].set_ylabel("Bit")
axes[0].set_yticks([0, 1])
axes[0].grid(True)

# Plot 2: Frequency hopping pattern
axes[1].step(
    bit_edges[:-1],
    hop_frequencies,
    where="post"
)
axes[1].set_title("2. Frequency Hopping Pattern")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_yticks(frequencies)
axes[1].grid(True)

# Plot 3: Transmitted signal
axes[2].plot(t, transmitted, linewidth=0.8)
axes[2].set_title("3. Transmitted FHSS Signal")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

# Plot 4: Received signal
axes[3].plot(t, received, linewidth=0.8)
axes[3].set_title("4. Received Signal with Noise")
axes[3].set_xlabel("Time (seconds)")
axes[3].set_ylabel("Amplitude")
axes[3].grid(True)

plt.tight_layout()
plt.show()
