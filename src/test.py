from src.data import Data
from skimage import io, filters, feature,transform
import matplotlib.pyplot as plt
import os

data = Data()
img = io.imread(os.path.join(os.path.join(data.CROP_DIR, '00000'), '00000_00001.ppm'),as_gray=True)
# edges = filters.sobel(img)
img=transform.resize(img,(32,32))
edges=feature.canny(img,sigma=1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(edges)
plt.show()
