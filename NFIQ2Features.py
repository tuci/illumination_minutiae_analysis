import numpy as np
import cv2
import scipy.ndimage as ndi
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class NFIQ2Features:
    def __init__(self, block_size=16, min_peaks=3, if_mask=True):
        self.block_size = block_size
        self.min_peaks = min_peaks
        self.mask = if_mask

    def block_view(self, A):
        """Divide image into non-overlapping blocks"""
        shape = (A.shape[0]//self.block_size, A.shape[1]//self.block_size, 
                 self.block_size, self.block_size)
        strides = (A.strides[0]*self.block_size, A.strides[1]*self.block_size, 
                   A.strides[0], A.strides[1])
        return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)

    def get_rotated_profile(self, B, theta):
        """
        Rotates the block B by angle theta so that ridges are horizontal, 
        then takes a vertical mean to create the profile.
        
        Args:
            B (np.ndarray): The 16x16 block.
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
        
        # Now that the ridges are aligned horizontally, the profile perpendicular 
        # to them is simply the average across the ridges (mean along the columns).
        # Mean across rows (axis=0) gives the profile perpendicular to the horizontal ridges.
        # Check: If ridges are horizontal, the profile is taken vertically (across rows).
        # profile = B_rotated.mean(axis=0) 
        
        # We need the profile *across* the ridges (the dimension H or W).
        # If the block size is 16x16, the result should be a 1D array of size 16.
        # If ridges are horizontal (0 deg), the profile should be mean(axis=1).
        profile = B_rotated.mean(axis=0) # Average along the ridge (rows) to get the profile across the block height

        # # --- 1. Calculate Block Center Coordinates (in image pixels) ---
        # center_x = H / 2
        # center_y = W / 2

        # # --- 2. Calculate Line Endpoints ---
        # # Length of the segment in pixels
        # line_length_ratio = 0.8
        # length = H * line_length_ratio 
        # half_length = length / 2

        # # Decompose the length into X and Y components based on the angle (theta)
        # # The orientation angle (theta) is measured with respect to the horizontal axis (X).
        # u = half_length * np.cos(theta)
        # v = half_length * np.sin(theta)

        # # Start point (x1, y1)
        # x1 = center_x - u
        # y1 = center_y - v
        # # End point (x2, y2)
        # x2 = center_x + u
        # y2 = center_y + v

        # # --- 3. Plot the Line Segment using Matplotlib ---
        # # Assuming 'ax' is your Matplotlib Axes object (e.g., from plt.subplots())
        # # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # # ax1.imshow(B, cmap='gray')
        # # ax1.plot([x1, x2], [y1, y2], color='red', linewidth=1.5)
        # # ax2.imshow(B_rotated, cmap='gray')
        # # ax3.plot(profile)
        # # plt.show()
        
        return profile

    def compute_ocl(self, img, mask=None):
        """Orientation Certainty Level (coherence of gradient field)"""
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        blocks_x = self.block_view(gx)
        blocks_y = self.block_view(gy)
        ocl_map = np.zeros(blocks_x.shape[:2])
        if mask is not None:
            blocks_mask = self.block_view(mask)

        for i in range(blocks_x.shape[0]):
            for j in range(blocks_x.shape[1]):
                # ignore background blocks
                if self.mask:
                    if np.sum(blocks_mask[i, j]) / blocks_mask[i, j].size <= 0.8:
                        continue
                Gx = blocks_x[i,j].ravel()
                Gy = blocks_y[i,j].ravel()
                J = np.array([[np.sum(Gx*Gx), np.sum(Gx*Gy)],
                            [np.sum(Gx*Gy), np.sum(Gy*Gy)]])
                w = np.linalg.eigvalsh(J)
                ocl_map[i,j] = (w[1]-w[0])/(w[1]+w[0]+1e-9)
        return ocl_map

    def compute_rvu(self, img, mask=None):
        """Ridge-Valley Uniformity (closer to NFIQ2)"""
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        blocks = self.block_view(img)
        rvu_map = np.zeros(blocks.shape[:2])
        if mask is not None:
            blocks_mask = self.block_view(mask)

        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                if self.mask:
                    # skip mostly background
                    if np.sum(blocks_mask[i, j]) / blocks_mask[i, j].size <= 0.8:
                        continue

                B = blocks[i, j].astype(np.float64)

                # orientation estimation via gradients
                Gx = gx[i*self.block_size:(i+1)*self.block_size,
                        j*self.block_size:(j+1)*self.block_size]
                Gy = gy[i*self.block_size:(i+1)*self.block_size,
                        j*self.block_size:(j+1)*self.block_size]
                
                theta = 0.5*np.arctan2(2*np.sum(Gx*Gy),
                                    np.sum(Gx**2 - Gy**2) + 1e-9)

                # True Orthogonal Projection via Rotation
                profile = self.get_rotated_profile(B, theta)

                # # project perpendicular to ridge flow
                # if abs(np.cos(theta)) > abs(np.sin(theta)):
                #     profile = B.mean(axis=0)  # vertical projection
                # else:
                #     profile = B.mean(axis=1)  # horizontal projection

                # smooth profile a bit
                profile = cv2.GaussianBlur(profile.reshape(-1, 1), (5, 1), 0).ravel()

                # find valleys (bright) and ridges (dark)
                peaks, _ = find_peaks(-profile, distance=6)   # ridges
                valleys, _ = find_peaks(profile, distance=6)  # valleys

                if len(peaks) > 0 and len(valleys) > 0:
                    ridge_mean = profile[peaks].mean()
                    valley_mean = profile[valleys].mean()
                    contrast = valley_mean - ridge_mean
                    # normalize by profile std to match NFIQ2 spirit
                    contrast /= (profile.std() + 1e-9)
                    rvu_map[i, j] = max(0, contrast)

        return rvu_map

    def compute_lcs(self, img, mask=None):
        """
        Local Clarity Score: checks for alternating ridge/valley profile
        - project block along ridge-perpendicular direction (simplified with Sobel)
        - count number/strength of peaks
        """
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        blocks = self.block_view(img)
        lcs_map = np.zeros(blocks.shape[:2])
        if mask is not None:
            blocks_mask = self.block_view(mask)
        
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                if self.mask:
                    # ignore background masks
                    if np.sum(blocks_mask[i, j]) / blocks_mask[i, j].size < 1.0:
                        continue
                B = blocks[i,j].astype(np.float64)
                # crude orientation: dominant gradient direction
                Gx = gx[i*self.block_size:(i+1)*self.block_size, 
                        j*self.block_size:(j+1)*self.block_size]
                Gy = gy[i*self.block_size:(i+1)*self.block_size, 
                        j*self.block_size:(j+1)*self.block_size]
                theta = 0.5*np.arctan2(2*np.sum(Gx*Gy), np.sum(Gx**2 - Gy**2)+1e-9)

                # True Orthogonal Projection via Rotation
                profile = self.get_rotated_profile(B, theta)
                
                # # project intensities along perpendicular to theta
                # if abs(np.cos(theta)) > abs(np.sin(theta)):
                #     profile = B.mean(axis=0)  # project along y
                # else:
                #     profile = B.mean(axis=1)  # project along x
                
                # detect peaks (valleys/ridges)
                peaks, _ = find_peaks(-profile, distance=2)  # ridges = dark = negative
                valleys, _ = find_peaks(profile, distance=2)
                
                if len(peaks) >= self.min_peaks and len(valleys) >= self.min_peaks:
                    contrast = profile[valleys].mean() - profile[peaks].mean()
                    lcs_map[i,j] = max(0, contrast)
        return lcs_map
