from src.data import Data
from skimage import io,filters, feature
import matplotlib.pyplot as plt
import os
data=Data()
img=io.imread(os.path.join(os.path.join(data.TRAIN_DIR,'00002'),'00000_00001.ppm'),as_gray=True)
# edges = filters.sobel(img)
edges=feature.canny(img,sigma=0.6)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(edges)
plt.show()
