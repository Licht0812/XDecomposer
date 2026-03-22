import argparse
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def broaden_pattern(pattern, two_theta_range=(5, 90), step_size=0.01, sigma=0.2):
    """
    Broaden the discrete peaks from pymatgen into a continuous spectrum using Gaussian Broadening.
    """
    x = np.arange(two_theta_range[0], two_theta_range[1], step_size)
    y = np.zeros_like(x)

    # Superimpose Gaussian distribution for each peak
    for angle, intensity in zip(pattern.x, pattern.y):
        y += intensity * gaussian(x, angle, sigma)
    
    # Normalize the intensity to [0, 1]
    if y.max() > 0:
        y /= y.max()
        
    return x, y

def main():
    parser = argparse.ArgumentParser(description="Compare Theoretical CIF XRD with Model Data")
    parser.add_argument("cif_path", type=str, help="Path to the .cif file")
    parser.add_argument("--wavelength", type=str, default="CuKa", help="Wavelength source (default: CuKa)")
    parser.add_argument("--sigma", type=float, default=0.2, help="Gaussian broadening sigma (default: 0.2)")
    parser.add_argument("--save_path", type=str, default="comparison_plot.png", help="Path to save the plot")
    
    # Optional: Compare with a raw data file (e.g., .npy or .txt)
    parser.add_argument("--compare_data", type=str, default=None, help="Path to experimental/model data file (optional)")
    
    args = parser.parse_args()

    # 1. Load Structure and Calculate Pattern
    print(f"Loading structure from {args.cif_path}...")
    try:
        structure = Structure.from_file(args.cif_path)
    except Exception as e:
        print(f"Error loading CIF: {e}")
        return

    xrd = XRDCalculator(wavelength=args.wavelength)
    pattern = xrd.get_pattern(structure, two_theta_range=(5, 90))
    
    print(f"Found {len(pattern.x)} peaks.")

    # 2. Broaden the pattern
    print("Applying Gaussian broadening...")
    x_theo, y_theo = broaden_pattern(pattern, sigma=args.sigma)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Discrete Peaks (Stem)
    plt.stem(pattern.x, pattern.y / 100.0, linefmt='k-', markerfmt=' ', basefmt=' ', label='Theoretical Peaks (Discrete)')
    
    # Plot Broadened Curve
    plt.plot(x_theo, y_theo, 'r-', linewidth=2, label=f'Theoretical Broadened (sigma={args.sigma})')

    # Plot Comparison Data if provided
    if args.compare_data:
        try:
            # Simple loader - adapt based on actual file format
            if args.compare_data.endswith('.npy'):
                data = np.load(args.compare_data)
                # Assume data matches the x-axis range or is just y-values
                if len(data.shape) == 1:
                    x_data = np.linspace(5, 90, len(data))
                    plt.plot(x_data, data, 'b--', label='Comparison Data')
                else:
                    print("Warning: .npy file shape not 1D, skipping plot.")
            else:
                print("Unsupported data format. Please implement specific loader.")
        except Exception as e:
            print(f"Error loading comparison data: {e}")

    plt.title(f"XRD Comparison: {args.cif_path}")
    plt.xlabel(r"2$\theta$ (degrees)")
    plt.ylabel("Intensity (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.save_path)
    print(f"Plot saved to {args.save_path}")

if __name__ == "__main__":
    main()
