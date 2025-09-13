import numpy as np
import matplotlib.pyplot as plt


# Importing necessary functions and parameters from my sampling_quantization.py file
from sampling_quantization import (
    signal_freq, duration, sampling_freq, num_bits, min_signal, max_signal,
    original_signal, sample_times, quantize
)

# ---- Noise parameters ----
mean = 0.0
std_dev = 0.1  

# This function adds Gaussian noise to a signal
def add_Gaussian_noise(signal, mean=0.0, std=0.1):
    mag = np.max(signal) - np.min(signal)
    noise = np.random.normal(mean, std * mag, size=len(signal))
    return signal + noise

# This function computes the Mean Squared Error (MSE)
def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))

# This function computes the Root Mean Square Error (RMSE)
def rmse(a, b):
    return float(np.sqrt(mse(a, b)))

# This function computes the Peak Signal-to-Noise Ratio (PSNR)
def psnr(max_signal_amp, mse_val):
    if mse_val <= 0:
        return float("inf")
    return 10.0 * np.log10((max_signal_amp ** 2) / mse_val)

def main():
    # creating original clean signal
    t_dense = np.linspace(0.0, duration, 1000, endpoint=False)
    clean_cont = original_signal(t_dense)
    
    # sampling the signal
    t_samp = sample_times(duration, sampling_freq)
    clean_samp = original_signal(t_samp)
    
    # adding Gaussian noise to the oringinal signal
    noisy_samp = add_Gaussian_noise(clean_samp, mean=mean, std=std_dev)
    
    # Create noisy continuous signal for visualization
    noisy_cont = add_Gaussian_noise(clean_cont, mean=mean, std=std_dev)
    
    # quantizing both clean and noisy sampled signals
    _, q_clean = quantize(clean_samp, num_bits, min_signal, max_signal)
    _, q_noisy = quantize(noisy_samp, num_bits, min_signal, max_signal)

    #----- Compute Error Metrics --------
    m = mse(q_clean, q_noisy)
    r = rmse(q_clean, q_noisy)
    peak = max(abs(min_signal), abs(max_signal)) 
    p = psnr(peak, m)
    print(f"MSE  = {m:.6f}")
    print(f"RMSE = {r:.6f}")
    print(f"PSNR = {p:.2f} dB")

    #----- Plot for Results--------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Original continuous signal vs noisy continuous signal
    ax1.plot(t_dense, clean_cont, label="Original continuous signal", linewidth=2, color='blue')
    ax1.plot(t_dense, noisy_cont, label="Noisy continuous signal", linewidth=1, color='red', alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Original vs Noisy Continuous Signal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    # Right plot: Clean quantized vs noisy quantized signals
    ax2.step(t_samp, q_clean, where="post", label="Quantized Original", linewidth=2, color='blue')
    ax2.step(t_samp, q_noisy, where="post", label="Quantized Noisy", linewidth=2, color='red', linestyle='--')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Original vs Noisy Quantized Signal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    plt.tight_layout()
    plt.savefig("Output/continuous_vs_quantized_noise.png", dpi=300)  
    plt.show()

    

if __name__ == "__main__":
    main()
