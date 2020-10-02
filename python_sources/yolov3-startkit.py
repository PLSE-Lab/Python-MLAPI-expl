#!/usr/bin/env python
# coding: utf-8

# ### YOLO Startkit using YOLOv3 Ultralytics
# Change your internet settings to On

# In[ ]:


get_ipython().system('git clone https://github.com/ultralytics/yolov3 ')


# In[ ]:


get_ipython().system('python3 /kaggle/working/yolov3/train.py --data /kaggle/input/maskdetection/yolo/obj.data --img-size 320 --epochs 1 --batch-size 16 --weights /kaggle/input/maskdetection/yolo/yolov3-spp-ultralytics.pt --cfg /kaggle/input/maskdetection/yolo/yolov3-mask-spp.cfg')


# In[ ]:


get_ipython().system('python3 /kaggle/working/yolov3/detect.py --names /kaggle/working/yolov3/data/coco.names --weights /kaggle/input/maskdetection/yolo/yolov3-spp-ultralytics.pt --cfg /kaggle/working/yolov3/cfg/yolov3-spp.cfg  --source /kaggle/working/yolov3/data/samples/bus.jpg --conf-thres 0.3 --iou-thres 0.6')


# In[ ]:


from PIL import Image
Image.open('/kaggle/working/output/bus.jpg')


# In[ ]:


get_ipython().system('python3 /kaggle/working/yolov3/detect.py --names /kaggle/input/maskdetection/yolo/obj.names --weights /kaggle/input/maskdetection/yolo/best.pt --cfg /kaggle/input/maskdetection/yolo/yolov3-mask-spp.cfg  --source /kaggle/input/maskdetection/yolo/images/test --conf-thres 0.3 --iou-thres 0.6')


# In[ ]:


from PIL import Image
Image.open('/kaggle/working/output/173.jpg')


# In[ ]:


get_ipython().system('python3 /kaggle/working/yolov3/test.py --cfg /kaggle/input/maskdetection/yolo/yolov3-mask-spp.cfg --weights /kaggle/input/maskdetection/yolo/best.pt --img 320 --augment --data /kaggle/input/maskdetection/yolo/obj.data')


# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/working/yolov3')


# In[ ]:


def plot_results(start=0, stop=0, bucket='', id=()):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3#training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                ax[i].plot(x, y, marker='.', label=Path(f).stem, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                if i in [5, 6, 7]:  # share train and val loss y axes
                    ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig('/kaggle/working/yolov3/results.png', dpi=200)


# In[ ]:


get_ipython().system('python3 -c "from utils import utf')
ils; utils.plot_results()"  # plot training results
Image(filename='/kaggle/working/yolov3/results.png', width=800)

