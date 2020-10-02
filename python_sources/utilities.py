import matplotlib.pyplot as plt
import numpy as np

def vis_training(hlist, start=1, size=[12,6]):
    
    tr_loss = []
    va_loss = []
    tr_acc = []
    va_acc = []
    
    for h in hlist:
        tr_loss += h.history['loss']
        va_loss += h.history['val_loss']
        tr_acc += h.history['accuracy']
        va_acc += h.history['val_accuracy']
        
    plt.figure(figsize = size)
    
    a = start
    b = len(tr_loss) + 1
    
    plt.subplot(1,2,1)
    plt.plot(range(a,b), tr_loss[a-1:], label='Training')
    plt.plot(range(a,b), va_loss[a-1:], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(range(a,b), tr_acc[a-1:], label='Training')
    plt.plot(range(a,b), va_acc[a-1:], label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.show()