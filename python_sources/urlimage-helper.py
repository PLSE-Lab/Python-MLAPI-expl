
"""def url_to_image(url, to_numpy = False):
    from PIL import Image
    import requests
    from io import BytesIO
    from piltonumpy_helper import pil_to_numpy
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    if to_numpy == True:
        img = pil_to_numpy(img)
        
    return img

def url_to_image2(url, to_numpy = False):
    from io import BytesIO
    import urllib.request as urllib
    from PIL import Image
    from piltonumpy_helper import pil_to_numpy
    
    img = Image.open(BytesIO(urllib.urlopen(url).read()))
    if to_numpy == True:
        img = pil_to_numpy(img)
    
    return img"""
    

def url_to_image(url, to_numpy = False):
    from skimage import io
    import cv2
    from PIL import Image
    
    image = io.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
    