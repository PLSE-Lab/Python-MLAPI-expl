from PIL import Image

base = Image.open('train/1_1.tif')
mask = Image.open('train/1_1_mask.tif')
res = base.convert('RGBA')

for x in range(base.width):
    for y in range(base.height):
        pixel = mask.getpixel((x, y))
        red, green, blue, _ = res.getpixel((x,y))

        if pixel == 255:
            res.putpixel((x, y), (int(255*0.7+red*0.3),green,blue))

res.save('mask.png', 'PNG')