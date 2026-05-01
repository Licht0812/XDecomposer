
import numpy as np
from skimage import restoration
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt
import matplotlib.pyplot as plt

def smooth_spectrum(spectrum, n=20):
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, spectrum)

def process_spectrum(ys):
    # ys is already 4501 points

    ## Smooth out noise
    ys = smooth_spectrum(ys)

    ## Normalize from 0 to 255
    ys = np.array(ys) - min(ys)
    ys = list(255*np.array(ys)/max(ys))

    # Subtract background
    background = restoration.rolling_ball(ys, radius=800)
    ys_no_bg = np.array(ys) - np.array(background)

    ## Normalize from 0 to 100
    ys_final = np.array(ys_no_bg) - min(ys_no_bg)
    ys_final = list(100*np.array(ys_final)/max(ys_final))

    return ys, background, ys_final

# Create a clean spectrum with some peaks
x = np.linspace(10, 80, 4501)
y = np.zeros(4501)
y[1000] = 100
y[2000] = 50
y[3000] = 75
y = gaussian_filter1d(y, 10) # broaden peaks

ys_raw, bg, ys_final = process_spectrum(y)

plt.figure(figsize=(10, 6))
plt.plot(x, ys_raw, label='Original (scaled to 255)')
plt.plot(x, bg, label='Rolling Ball BG')
plt.plot(x, ys_final, label='Final (scaled to 100)')
plt.legend()
plt.savefig('debug_bg.png')
print("Debug plot saved to debug_bg.png")

print(f"Max difference between original and final: {np.max(np.abs(np.array(ys_raw)/2.55 - np.array(ys_final)))}")
