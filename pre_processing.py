import cv2, math
import numpy as np
import scipy.ndimage as ndi
from NFIQ2Features import NFIQ2Features

# define pre-processing operations

# process image-set
def process_image_set(imageset, histEq=True, target_ridge_distance=9, min_block=16, max_block=96):
    # loop over images
    scales = []
    for image in imageset:
        image, image_scale = process_image(image, histEq=histEq, desired_ridge_period=target_ridge_distance, 
                              min_block=min_block, max_block=max_block)
        scales.append(image_scale)

    scale = np.nanmedian(scales)
    if target_ridge_distance is not None:
        imageset = normalise_images_to_ridge_period(imageset, scale)

    return imageset

def process_image(imageSet, histEq, desired_ridge_period, min_block, max_block):
    # convert image to gray
    imageLAB = cv2.cvtColor(imageSet.image, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(imageLAB)
    imageSet.imageGray = L
    
    if histEq:
        imageSet.imageGray = apply_clahe(L)
    
    # extract roi
    roi, roi_mask, roi_coords = extract_roi(L, imageSet.mask)
    roiC, _, (y_min, y_max, x_min, x_max) = extract_roi(imageSet.image, imageSet.mask)
    # grayscale ROI
    imageSet.roi = roi
    imageSet.roiMask = roi_mask
    imageSet.roiCoords = (y_min, x_min)
    # colored ROI(s)
    imageSet.roiColor = roiC
    # if minutiae annotations are given
    if hasattr(imageSet, 'annotated'):
        roiCA, _, _ = extract_roi(imageSet.annotated, imageSet.mask)
        imageSet.roiColorA = roiCA

    # normalise roi to ridge frequency
    if desired_ridge_period is not None:
        median_ridge_period = compute_ridge_period_image(imageSet.imageGray, imageSet.mask, min_block, max_block)
        image_scale = desired_ridge_period / median_ridge_period
    else:
        image_scale = None
    
    return imageSet, image_scale

def extract_roi(image, mask, margin=10):
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 and len(ys) == 0:
        raise ValueError('Empty Mask -- Nothing to crop!!')
    # bounding box
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    # add padding to right and bottom 
    y_max = min(mask.shape[0], y_max + margin)
    x_max = min(mask.shape[1], x_max + margin)

    # crop region
    cropped_image = image[y_min-margin:y_max, x_min-margin:x_max]
    cropped_mask = mask[y_min-margin:y_max, x_min-margin:x_max]

    return cropped_image, cropped_mask, (y_min, y_max, x_min, x_max)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=6.0)
    return clahe.apply(image)

def powers_of_two_between(low, high):
    start = math.ceil(math.log2(low))
    end = math.floor(math.log2(high))

    return [2**i for i in range(start, end + 1)]

# normalise roi to 8-10px ridge period
def normalise_images_to_ridge_period(imageset, median_scale):
    for image in imageset:
        # scale the roi and mask accordingly
        interpolation = cv2.INTER_AREA if median_scale < 1 else cv2.INTER_CUBIC
        image.normalisedROI = cv2.resize(image.roi, (None, None), fx=median_scale, fy=median_scale, 
                                         interpolation=interpolation)
        image.normalisedROIMask = cv2.resize(image.roiMask.astype(np.uint8), (None, None), fx=median_scale, fy=median_scale, 
                                             interpolation=cv2.INTER_LINEAR).astype(bool)
        image.normalisedROIC = cv2.resize(image.roiColor, (None, None), fx=median_scale, fy=median_scale, 
                                          interpolation=cv2.INTER_AREA)
        image.normalisedROICA = cv2.resize(image.roiColorA, (None, None), fx=median_scale, fy=median_scale, 
                                           interpolation=cv2.INTER_AREA)
        image.median_scale = median_scale
        
    return imageset

def compute_ridge_period_image(image, mask, min_block, max_block):
    block_range = np.linspace(min_block, max_block, num=10, dtype=int)

    arps = []
    for block_size in block_range:
        # average ridge period for the image
        arp = compute_ARP_image(image, mask, block_size)
        arps.append(arp)

    median_period = np.nanmedian(arps)

    return median_period

# compute average ridge period for image
def compute_ARP_image(image, mask, block_size):
    # detect reliable blocks for ridge period estimation using OCL 
    nfiq = NFIQ2Features(block_size=block_size)
    ocl = nfiq.compute_ocl(image, mask)
    windows = nfiq.block_view(image)
    # compute orientaion consistency threshold
    ocl_thresh = np.percentile(ocl.flatten(), q=85)

    # get blocks above threshold
    goodY, goodX = np.where(ocl > ocl_thresh)
    if len(goodY) == 0: # no eligible window found
        return np.nan
    
    average_ridge_period = 0.0
    for idy, idx in zip(goodY, goodX):
        block = windows[idy, idx]
        block_period = compute_ridge_period_window(block)
        if block_period is None:
            print('No period found')
        average_ridge_period += block_period

    # compute scaling factor
    average_ridge_period /= len(goodY)

    return average_ridge_period
    
# compute ridge period per window
def compute_ridge_period_window(window):
    # compute gradients
    Gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)

    # orientation angle
    theta = 0.5*np.arctan2(2*np.sum(Gx*Gy), np.sum(Gx**2 - Gy**2) + 1e-9)

    # rotated profile
    profile = get_rotated_profile(window, theta)

    # smooth the profile
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (5, 1), 0).ravel()

    # compute window ridge period
    period, conf = get_profile_period_fft(profile)

    return period

def get_rotated_profile(B, theta):
    """
    Rotates the block B by angle theta so that ridges are horizontal, 
    then takes a vertical mean to create the profile.
    
    Args:
        B (np.ndarray): The ridge window.
        theta (float): The local ridge orientation angle (in radians).
        
    Returns:
        np.ndarray: The 1D profile perpendicular to the original ridge flow.
    """
    H, W = B.shape
    
    # We want the ridges to be vertical (perpendicular to horizontal projection), 
    # so we rotate by (theta + pi/2) to make the perpendicular direction vertical.
    # However, standard practice is to align the ridges themselves horizontally 
    # (i.e., rotate by -theta), and then project vertically (mean(axis=1)). 
    # Let's use the latter, simpler rotation target: make ridges horizontal.
    
    # Rotation angle: Rotate by -theta so that the ridge direction becomes 0 degrees (horizontal)
    rotation_angle_deg = np.rad2deg(theta)
    
    # Use continuous interpolation (order=1 for linear) to rotate the block.
    # This is much slower than a simple mean but mathematically correct.
    B_rotated = ndi.rotate(
        B, 
        angle=rotation_angle_deg, 
        reshape=False, 
        order=1, 
        mode='nearest' # Use nearest to avoid boundary artifacts
    )
    
    # We need the profile *across* the ridges (the dimension H or W).
    # If the block size is 16x16, the result should be a 1D array of size 16.
    # If ridges are horizontal (0 deg), the profile should be mean(axis=1).
    profile = B_rotated.mean(axis=0) # Average along the ridge (rows) to get the profile across the block height
    
    return profile

def get_profile_period_fft(profile):
    p = profile.astype(np.float32)
    # detend and Hann to reduce leakage
    n = len(p)
    t = np.arange(n, dtype=np.float32)
    p = p - np.poly1d(np.polyfit(t, p, 1))(t)
    p *= np.hanning(n)

    # comput fft
    S = np.fft.rfft(p)
    mag = np.abs(S)
    mag[:2] = 0 # reduce DC and very low frequency

    # find the dominant frequency
    k = int(np.argmax(mag))
    # if no valid frequency found
    if k <= 0.0:
        return None, 0.0

    f = k / n # cycles per pixel
    period = 1.0 / f

    # confidence - peak prominence and median spectrum
    # ratio between the peak magnitude and median magnitude
    conf = float((mag[k] + 1e-6) / (np.median(mag[1:]) + 1e-6))

    return period, conf        














