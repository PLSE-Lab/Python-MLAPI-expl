import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature


# Generate noisy image of a square
im = np.zeros((128, 128))
im[32:-32, 32:-32] = 1

im = ndi.rotate(im, 15, mode='constant')
im = ndi.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)
# define scharr filters
scharr = zeros(3,3,2);
scharr[:,:,1] = [3, 10, 3, 0, 0, 0, -3, -10, -3];
scharr[:,:,2] = [3, 0, -3, 10, 0, -10, 3, 0, -3];
# Compute the Scharr filter
edges = zeros(128,128,2);
for i in range(0,2):
    edges[:,:,i] = filter2(scharr(:,:,i), im);
end

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges(:,:,1), cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Scharr filter, X filtered, fontsize=20)

ax3.imshow(edges(:,:,2), cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Scharr filter, Y filtered, fontsize=20)

fig.tight_layout()

plt.show()
plt.close()