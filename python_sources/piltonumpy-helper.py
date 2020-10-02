def pil_to_numpy(img):
    import numpy as np
    from PIL import Image
    img = img.convert('RGB')
    return np.array(img)

