
import numpy as np
import matplotlib.pyplot as plt



np.random.seed(42)

# 1. Parameters
constraint_length = 3
memory = constraint_length - 1

# Generator polynomials: 111 and 101 in binary
generators = [0b111, 0b101]

num_states = 2 ** memory
noise_probability = 0.08

# 2. Original binary message
data = np.array([
    1, 1, 0, 1, 0, 0, 1, 1
], dtype=int)

# Append zeros to return encoder to state 0
input_bits = np.concatenate([
    data, np.zeros(memory, dtype=int)
])


# 3. Convolutional encoder
def convolutional_encode(bits):
    state = 0
    encoded = []

    for bit in bits:
        # Register: current input + previous memory bits
        register = (int(bit) << memory) | state

        # Generate output bits using XOR
        for generator in generators:
            output = bin(register & generator).count("1") % 2
            encoded.append(output)

        # Keep the newest memory bits
        state = register & (num_states - 1)

    return np.array(encoded, dtype=int)


# 4. Build trellis transitions
def next_state_output(state, bit):
    register = (int(bit) << memory) | state

    outputs = []
    for generator in generators:
        output = bin(register & generator).count("1") % 2
        outputs.append(output)

    next_state = register & (num_states - 1)

    return next_state, np.array(outputs)


# 5. Viterbi decoder
def viterbi_decode(received):
    received_pairs = received.reshape(-1, len(generators))
    steps = len(received_pairs)

    # Start in state 0
    infinity = float("inf")
    path_metrics = np.full(num_states, infinity)
    path_metrics[0] = 0

    # Store the best predecessor and input bit
    predecessors = np.full(
        (steps, num_states), -1, dtype=int
    )
    decisions = np.full(
        (steps, num_states), -1, dtype=int
    )

    for t in range(steps):
        new_metrics = np.full(num_states, infinity)

        for state in range(num_states):
            if path_metrics[state] == infinity:
                continue

            for bit in [0, 1]:
                next_state, expected = next_state_output(
                    state, bit
                )

                # Hamming distance: number of mismatches
                branch_metric = np.sum(
                    expected != received_pairs[t]
                )

                candidate = (
                    path_metrics[state] + branch_metric
                )

                if candidate < new_metrics[next_state]:
                    new_metrics[next_state] = candidate
                    predecessors[t, next_state] = state
                    decisions[t, next_state] = bit

        path_metrics = new_metrics

    # Encoder was terminated with zeros; choose state 0
    state = 0
    decoded = []

    for t in range(steps - 1, -1, -1):
        bit = decisions[t, state]
        decoded.append(bit)
        state = predecessors[t, state]

    decoded.reverse()
    return np.array(decoded, dtype=int)


# 6. Encode the message
encoded = convolutional_encode(input_bits)

# 7. Simulate a binary symmetric channel
noise = (
    np.random.random(len(encoded)) < noise_probability
).astype(int)

received = encoded ^ noise

# 8. Decode using Viterbi
decoded = viterbi_decode(received)

# Remove termination bits
decoded_data = decoded[:-memory]

# 9. Display results
print("=" * 55)
print("CONVOLUTIONAL ENCODING AND VITERBI DECODING")
print("=" * 55)

print("Original data:       ", data)
print("Encoder input:       ", input_bits)
print("Encoded bitstream:   ", encoded)
print("Received bitstream:  ", received)
print("Decoded bitstream:   ", decoded)
print("Recovered data:      ", decoded_data)
print("Channel bit errors:  ", np.sum(encoded != received))
print("Decoding errors:     ", np.sum(data != decoded_data))
print("Successful recovery: ",
      np.array_equal(data, decoded_data))


# 10. Plot the results
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

signals = [
    (data, "1. Original Data"),
    (encoded, "2. Convolutionally Encoded Data"),
    (received, "3. Received Data After Channel Noise"),
    (decoded_data, "4. Viterbi Decoded Data")
]

for ax, (signal, title) in zip(axes, signals):
    ax.step(
        np.arange(len(signal)),
        signal,
        where="mid"
    )
    ax.set_title(title)
    ax.set_ylabel("Bit")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True)

axes[-1].set_xlabel("Bit index")

plt.tight_layout()
plt.show()
