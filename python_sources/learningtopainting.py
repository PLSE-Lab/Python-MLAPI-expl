#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/hzwer/LearningToPaint.git')


# In[2]:


cd LearningToPaint/


# In[ ]:


get_ipython().system('wget "https://drive.google.com/uc?export=download&id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4" -O renderer.pkl')


# In[ ]:


get_ipython().system('mkdir data')


# In[ ]:


cd data


# In[ ]:


get_ipython().system('pip install gdown')


# In[ ]:


get_ipython().system('gdown https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM')


# In[ ]:


get_ipython().system('unzip img_align_celeba.zip')


# In[ ]:


get_ipython().system('rm img_align_celeba.zip')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


cd ..


# In[ ]:


get_ipython().system('pip install opencv-python==3.4.0.14')


# In[ ]:


get_ipython().system('pip install tensorboardX')


# In[ ]:


get_ipython().system('pip install https://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl')


# In[ ]:


get_ipython().system('python3 baseline/train.py --max_step=40 --debug --batch_size=96')


# In[ ]:


cd baseline/


# In[ ]:


get_ipython().run_cell_magic('writefile', 'test.py', 'import cv2\nimport torch\nimport numpy as np\nimport argparse\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nfrom DRL.actor import *\nfrom Renderer.stroke_gen import *\nfrom Renderer.model import *\n\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\nwidth = 128\n\nparser = argparse.ArgumentParser(description=\'Learning to Paint\')\nparser.add_argument(\'--max_step\', default=40, type=int, help=\'max length for episode\')\nparser.add_argument(\'--path\', default=\'./model/Paint-run1/\', type=str, help=\'Actor model path\')\nargs = parser.parse_args()\n\nT = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)\n\ncoord = torch.zeros([1, 2, width, width])\nfor i in range(width):\n    for j in range(width):\n        coord[0, 0, i, j] = i / (width - 1.)\n        coord[0, 1, i, j] = j / (width - 1.)\ncoord = coord.to(device) # Coordconv\n\nDecoder = FCN()\nDecoder.load_state_dict(torch.load(\'./renderer.pkl\'))\n\ndef decode(x, canvas): # b * (10 + 3)\n    x = x.view(-1, 10 + 3)\n    stroke = 1 - Decoder(x[:, :10])\n    stroke = stroke.view(-1, 128, 128, 1)\n    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)\n    stroke = stroke.permute(0, 3, 1, 2)\n    color_stroke = color_stroke.permute(0, 3, 1, 2)\n    stroke = stroke.view(-1, 5, 1, 128, 128)\n    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)\n    for i in range(5):\n        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]\n    return canvas\n\nimg = cv2.imread(\'./image/test.png\', cv2.IMREAD_COLOR)\nimg = cv2.resize(img, (width, width))\nimg = np.transpose(img, (2, 0, 1))\nimg = torch.tensor(img).to(device).reshape(1, -1, width, width).float() / 255.\nactor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13\nactor.load_state_dict(torch.load(args.path + \'/actor.pkl\'))\nactor = actor.to(device).eval()\nDecoder = Decoder.to(device).eval()\n\ncanvas = torch.zeros([1, 3, width, width]).to(device)\n\nfor i in range(args.max_step):\n    stepnum = T * i / args.max_step\n    actions = actor(torch.cat([canvas, img, stepnum, coord], 1))\n    canvas = decode(actions, canvas)\n    print(\'step {}, L2Loss = {}\'.format(i, ((canvas - img) ** 2).mean()))\n    output = canvas[0].detach().cpu().numpy()\n    output = np.transpose(output, (1, 2, 0))\n    cv2.imwrite(\'image/generated\'+str(i)+\'.png\', (output * 255).astype(\'uint8\'))')


# In[ ]:


cd ..


# In[ ]:


get_ipython().system("python3 baseline/test.py --max_step=10000 --path='/kaggle/working/LearningToPaint/model/Paint-run1/'")


# In[ ]:


cd image/


# In[ ]:


get_ipython().system('apt install ffmpeg -y')


# In[ ]:


get_ipython().system('ffmpeg -r 10 -f image2 -i generated%d.png -vf scale="720:-1" -y video.mp4 ')


# In[ ]:




