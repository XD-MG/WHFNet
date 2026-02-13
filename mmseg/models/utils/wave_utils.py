import numpy as np
import pywt
import torch

def channel_normalize(image):
    result = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]): 
        channel = image[i, :, :]
        if np.max(channel) > np.min(channel): 
            normalized = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
            result[i, :, :] = np.uint8(normalized)
    return result


def wavelet_transform_chan(input_tensor, log_transform=False, wavelet='haar'):
    """
    wavelet transform for each channel
    """
    B, C, H, W = input_tensor.shape
    if len(input_tensor.shape) != 4:
        raise ValueError("Input tensor must be a 4D tensor.")
    
    low_freq_tensors = []
    high_freq_tensors = []
    for b in range(B):
        sample = input_tensor[b].cpu().numpy()
        if log_transform:
            sample = np.abs(sample) + 1e-6
            sample = np.log1p(sample)
        coeffs = pywt.dwt2(sample, wavelet, axes=(-2, -1))
        LL, (LH, HL, HH) = coeffs
        
        LL_upsampled = pywt.idwt2((LL, (None, None, None)), wavelet, axes=(-2, -1))
        H_merge_upsampled = pywt.idwt2((np.zeros_like(LL), (LH, HL, HH)), wavelet, axes=(-2, -1))
        LL_normalized = channel_normalize(LL_upsampled)
        HH_normalized = channel_normalize(H_merge_upsampled)

        low_freq_ = torch.from_numpy(LL_normalized).float().unsqueeze(0)
        high_freq_ = torch.from_numpy(HH_normalized).float().unsqueeze(0)

        low_freq_tensors.append(low_freq_)
        high_freq_tensors.append(high_freq_)

    low_freq_batch = torch.cat(low_freq_tensors, dim=0).to(input_tensor.device)
    high_freq_batch = torch.cat(high_freq_tensors, dim=0).to(input_tensor.device)    
    return low_freq_batch, high_freq_batch