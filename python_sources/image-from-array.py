from PIL import Image
import numpy as np
def train_to_image_and_save_train_val_txt():
    lines = [line.rstrip('\n') for line in open('train.csv')]
    train_header = lines.pop(0)
    train_val_arr = []
    img_id = 0
    for img_arr in lines:
        arr = img_arr.split(',')
        label = arr[0]
        new_arr = results = map(int, np.delete(arr,[0]))
        train_val_arr.append(str(img_id)+".jpg "+label)
        img = Image.new("L", (28, 28), "white")
        img.putdata(new_arr)
        img.save('train/'+str(label)+'/'+str(img_id)+'.jpg')
        img_id +=1
    with open('img_and_train.txt', 'w') as f:
        for s in train_val_arr:
            f.write(s + '\n')
def test_to_image():
    test_lines = [line.rstrip('\n') for line in open('test.csv')]
    test_header = test_lines.pop(0)
    test_img_id = 0
    for img_arr in test_lines:
        test_arr = results = map(int, img_arr.split(','))
        img = Image.new("L", (28, 28), "white")
        img.putdata(test_arr)
        img.save('test/'+str(test_img_id)+'.jpg')
        test_img_id +=1