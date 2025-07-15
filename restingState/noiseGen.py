import numpy as np

def calculateDepthScaling(layers):
    original_array = np.array([1, 1.5, 2.1, 2.7, 3, 3.2, 3.8, 3.3, 3.2, 3.5, 3.2, 5.2, 6]) # estimation based on Koopmans et al 2011 Figure 6
        
    pad_length = len(original_array)  # Pad with the same length as the original array
    padded_array = np.pad(original_array, (pad_length, pad_length))
    fft_result = np.fft.fft(padded_array)

    truncated_fft = np.zeros_like(fft_result)
    truncated_fft[:layers] = fft_result[:layers]

    downsampled_array = np.real(np.fft.ifft(truncated_fft))
    centered_downsampled = downsampled_array[pad_length:pad_length + len(original_array)]
    print(centered_downsampled[:layers])

def calculateDepthScaling_lin(layers, origArray=True):
    """
    Linearly interpolate the original depth estimates
    to 'layers' equally spaced values.
    """
    # Original depth estimates (Koopmans et al. 2011, Fig.6)
    # original_array = np.array([3, 3.7, 3.5])
    if origArray:
        original_array = np.array([1, 1.5, 2.1, 2.7, 3, 3.2, 3.8, 3.3, 3.2, 3.5, 3.2, 5.2, 6]) # estimation based on Koopmans et al 2011 Figure 6
    else:
        original_array = np.array([3, 3.7, 3.5])

    
    # Parameterize original points on [0,1]
    x_orig = np.linspace(0, 1, len(original_array))
    # New sample locations
    x_new = np.linspace(0, 1, layers)
    
    # Interpolate
    depth_scaling = np.interp(x_new, x_orig, original_array)
    print(depth_scaling/np.mean(depth_scaling))


calculateDepthScaling_lin(27)
calculateDepthScaling_lin(27, origArray=False)

print((9-3)//2)