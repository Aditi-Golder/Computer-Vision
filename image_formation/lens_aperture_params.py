import numpy as np
import matplotlib.pyplot as plt

# This function computes the image distance zi given focal length f and object distance z0
def thin_lens_zi(f_mm, z0_mm):
    z0 = np.array(z0_mm, dtype=np.float64)
    f = float(f_mm)
    zi = f * z0 / (z0 - f)
    return zi

# This function plots zi vs z0 for various focal lengths
def plot_zi_vs_z0():
    f_list = [3.0, 9.0, 50.0, 200.0] 
    plt.figure(figsize=(7, 5))
    for f in f_list:
        z0_min = 1.1 * f
        z0_max = 1.0e4
        step = 0.25  
        z0 = np.arange(z0_min, z0_max + step, step, dtype=np.float64)
        zi = thin_lens_zi(f, z0)
        plt.loglog(z0, zi, label=f"f = {f:g} mm")
        plt.axvline(f, linestyle="--")

    plt.xlabel(r"Object distance $z_o$ (mm)")
    plt.ylabel(r"Image distance $z_i$ (mm)")
    plt.ylim(0, 3000)
    plt.title("Thin Lens Law: $z_i$ vs $z_o$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Output/thin_lens_law.png")  
    plt.show()

# This function plots aperture diameter D vs focal length f for real lenses
def plot_D_vs_f():
    plt.figure(figsize=(10, 6))
    
    # Real lenses data with labels and colors
    real_lenses = [
        (24, 1.4, "24mm f/1.4", "red"),
        (50, 1.8, "50mm f/1.8", "blue"), 
        (70, 2.8, "70mm f/2.8", "green"),
        (200, 2.8, "200mm f/2.8", "orange"),
        (400, 2.8, "400mm f/2.8", "purple"),
        (600, 4.0, "600mm f/4.0", "brown")
    ]
    
    for f, n, label, color in real_lenses:
        D = f / n
        plt.scatter(f, D, color=color, s=80, zorder=5, label=label)
    
    plt.xlabel("Focal length f (mm)")
    plt.ylabel("Aperture diameter D (mm)")
    plt.title("Real Camera Lenses: Aperture Diameter vs Focal Length")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig("Output/real_lenses_aperture.png", bbox_inches='tight')
    plt.show()

# This for answering the question about max apertures of real lenses
def print_real_lens_max_apertures():
    data = [
        ("24 mm f/1.4", 24.0, 1.4),
        ("50 mm f/1.8", 50.0, 1.8),
        ("70–200 mm f/2.8 (70)", 70.0, 2.8),
        ("70–200 mm f/2.8 (200)", 200.0, 2.8),
        ("400 mm f/2.8", 400.0, 2.8),
        ("600 mm f/4.0", 600.0, 4.0),
    ]
    # This is the formula
    print("Max aperture diameters D = f / N:")
    for name, f, N in data:
        D = f / N
        print(f"  {name:<30s} -> D = {D:.2f} mm")

def main():
    plot_zi_vs_z0()
    plot_D_vs_f()
    print_real_lens_max_apertures()

if __name__ == "__main__":
    main()
