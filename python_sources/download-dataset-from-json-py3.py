# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys, os, multiprocessing, urllib3, csv
from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def ParseData(data_file):
  key_url_list = []
  j = json.load(open(data_file))
  images = j['images']
  for item in images:
    url = item['url']
    id_ = item['id'].split('.')[0]
    extention = item['id'].split('.')[-1]
    if 'train' in data_file or 'val' in data_file:
        label = item['class']
        id_= "{}_{}".format(label, id_)
    key_url_list.append((id_, url, extention))
  return key_url_list

def DownloadImage(key_url):
  out_dir = sys.argv[2]
  (key, url,extention) = key_url
  filename = os.path.join(out_dir, key+'.'+extention)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    #print('Trying to get %s.' % url)
    http = urllib3.PoolManager(timeout=10.0)
    response = http.request('GET', url)
    image_data = response.data
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s %s' % (key,url))
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return


def Run():
  if len(sys.argv) != 3:
    print('Syntax: %s <train|val|test.json> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=12)

  with tqdm(total=len(key_url_list)) as t:
    for _ in pool.imap_unordered(DownloadImage, key_url_list):
      t.update(1)


if __name__ == '__main__':
  Run()

# Any results you write to the current directory are saved as output.