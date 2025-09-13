import numpy as np
import matplotlib.pyplot as plt

# ---- Global parameters  ----
signal_freq = 5.0   
duration = 2.0      
sampling_freq = 8.0 
num_bits = 3        
min_signal = -1.0
max_signal =  1.0

# This function will generate the original continuous signal
def original_signal(t):
    f = signal_freq
    t = np.asarray(t, dtype=np.float64)
    return np.sin(2 * np.pi * f * t)

# This function will generate the sample times
def sample_times(duration, fs):
    n = int(fs * duration)
    return np.linspace(0.0, duration, n, endpoint=False)

# This function will quantize the samples
def quantize(samples, bits, smin, smax):
    #although in the given assignment formula it says (n-1) but to correctly calculate the levels I guess need to use 2^bits
    levels = int(2 ** bits) 
    q_s = np.round((samples - smin) / (smax - smin) * (levels - 1)).astype(int)
    q_s = np.clip(q_s, 0, levels - 1)
    q_v = smin + q_s * (smax - smin) / (levels - 1)
    return q_s, q_v

def main():
    # 1) Doing Continuous signal
    t_dense = np.linspace(0.0, duration, 1000, endpoint=False)
    cont = original_signal(t_dense)
    # 2) Doing Sampling
    t_samp = sample_times(duration, sampling_freq)
    samp = original_signal(t_samp)

    # 3) Doing Quantize
    q_idx, q_val = quantize(samp, num_bits, min_signal, max_signal)

    #----- Plot for Results--------
    plt.figure(figsize=(9, 4))
    plt.plot(t_dense, cont, label="Continuous signal")
    plt.step(t_samp, q_val, where="post", label=f"Quantized ({num_bits} bits)", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sampling & Quantization of a Sinusoid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Output/sampling_quantization.png", dpi=300)
    plt.show()

    # 5) Textual guidance for the report
    nyquist = 2.0 * signal_freq
    print(f"Signal frequency: {signal_freq:.2f} Hz")
    print(f"Sampling frequency: {sampling_freq:.2f} Hz  (Nyquist = {nyquist:.2f} Hz)")
    if sampling_freq < nyquist:
        print(">> Sampling is below Nyquist — expect aliasing and shape distortion.")
    else:
        print(">> Sampling meets/exceeds Nyquist; increasing fs further improves waveform fidelity.")
    print("To minimize error: (i) increase sampling rate, (ii) apply anti-alias low-pass filtering before sampling, "
          "(iii) increase quantization bit-depth, (iv) use dithering for better perceptual quality when appropriate.")

if __name__ == "__main__":
    main()
