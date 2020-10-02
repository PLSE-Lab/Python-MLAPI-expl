def visualize_convolutions():
    '''
    This function takes pictures in jpg of cats lol and using pre-defined kernels
    convert them in similar manner to what convolutional neural network does.
    We omit the training process and just show how different filters enhance certain
    characteristics of the picture. 
    
    packages: numpy, PIL, matplotlib.pyplot
    
    '''
    # convulving_matrix is the actual function to apply filters on picture
    def convulving_matrix(input_matrix, conv_kernel, stride=(1, 1), pad_method='same', bias=1):
    
        input_h, input_w, input_d = input_matrix.shape[0], input_matrix.shape[1], input_matrix.shape[2]
        kernel_h, kernel_w, kernel_d = conv_kernel.shape[0], conv_kernel.shape[1], conv_kernel.shape[2]
        stride_h, stride_w = stride[0], stride[1]
        
        if pad_method == 'same':
            # same is the method to returns pciture of the same size
            # so we are zero-padding around it
            output_h = int(np.ceil(input_matrix.shape[0] / float(stride[0])))
            output_w = int(np.ceil(input_matrix.shape[1] / float(stride[1])))
            output_d = input_d
            output = np.zeros((output_h, output_w, output_d))
            
            pad_h = max((output_h - 1) * stride[0] + conv_kernel.shape[0] - input_h, 0)
            pad_h_offset = int(np.floor(pad_h/2))  
            pad_w = max((output_w - 1) * stride[1] + conv_kernel.shape[1] - input_w, 0)
            pad_w_offset = int(np.floor(pad_w/2))
                               
            padded_matrix = np.zeros((output_h + pad_h, output_w + pad_w, input_d))
            
            for l in range(input_d):
                for i in range(input_h):
                    for j in range(input_w):
                        padded_matrix[i + pad_h_offset, j + pad_w_offset, l] = input_matrix[i, j, l]
            
            for l in range(output_d):
                for i in range(output_h):
                    for j in range(output_w):
                        curr_region = padded_matrix[i*stride_h : i*stride_h + kernel_h, j*stride_w : j*stride_w + kernel_w, l]
                        output[i, j, l] = (conv_kernel[..., l] * curr_region).sum()
                        
        elif pad_method == 'valid':
            
            output_h = int(np.ceil((input_matrix.shape[0] - kernel_h + 1) / float(stride[0])))
            output_w = int(np.ceil((input_matrix.shape[1] - kernel_w + 1) / float(stride[1])))
            output = np.zeros((output_h, output_w, layer+1))
            
            for l in range(layer + 1):
                for i in range(output_h):
                    for j in range(output_w): 
                        curr_region = input_matrix[i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w, l]
                        output[i, j, l] = (conv_kernel[..., l] * curr_region).sum()
    
        output = np.sum(output, axis=2) + bias
        
        return output  
    
    # convert_pic_to_array is used to convert 3dim rgb picture to 1dim greyscale
    def convert_pic_to_array(cat_number):
        from PIL import Image
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
        return rgb2gray(np.array(Image.open('../input/cat{}.jpg'.format(cat_number)).convert('RGB')))

    # HERE WE GO
    import numpy as np
    
    cats_matrix = np.zeros((10, 400, 600, 1))
    for i in range(1, 11):
        cats_matrix[i-1, :, :, 0] = convert_pic_to_array(i)
    
    global cats_in_disguise
    cats_in_disguise = np.zeros(shape=(10, 400, 600, 10))
    cats_in_disguise[:,:,:,0] = cats_matrix[:, :, :, 0]
    
    k1 = np.array([ 1,  2, 0,  -2, -1, 
                    4,  8, 0,  -8, -4, 
                    6, 12, 0, -12, -6, 
                    4,  8, 0,  -8, -4, 
                    1,  2, 0,  -2, -1])
    k2 = np.array([2, 2, 3, 2, 2, 
                   1, 1, 2, 1, 1, 
                   0, 0, 0, 0, 0, 
                   -1, -1, -2, -1, -1, 
                   -2, -2, -4, -2, -2])
    k3 = np.array([2, 1, 0, -1, -2, 
                   2, 1, 0, -1, -2, 
                   4, 2, 0, -2, -4, 
                   2, 1, 0, -1, -2, 
                   2, 1, 0, -1, -2])
    k4 = np.array([9, 9, 9, 9, 9, 
                   9, 5, 5, 5, 9, 
                   -7, -3, 0, -3, -7, 
                   -7, -3, -3, -3, -7, 
                   -7, -7, -7, -7, -7])
    k5 = np.array([9, 9, -7, -7, -7, 
                   9, 5, -3, -3, -7, 
                   9, 5, 0, -3, -7, 
                   9, 5, -3, -3, -7, 
                   9, 9, -7, -7, -7])  
    k6 = np.array([1,   4,   6,   4,   1,
                   2,   8,  12,   8,   2,
                   0,   0,   0,   0,   0,
                   -2,  -8, -12,  -8,  -2,
                   -1,  -4,  -6,  -4,  -1])
    k7 = np.array([0, 0, -1, 0, 0,
                   0, 0,  -1,  0, 0,
                   -1, -1,  8, -1, -1,
                   0, 0, -1,  0, 0,
                   0, 0, -1, 0, 0])
    k8 = np.array([-1, -1, -1, -1, -1,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1])
    k9 = np.array([-5,-4, 0, 4, 5,
                  -8, -10, 0, 10, 8,
                  -10, -20, 0, 20,10,
                  -8, -10, 0, 10, 8,
                  -5, -4, 0, 4, 5])

    kernels = np.vstack((k1, k2, k3, k4, k5, k6, k7, k8, k9))
    
    max_k = 9
    
    for i in range(1, max_k+1):
        kernel_s = np.sqrt(kernels[i-1].shape[0]).astype(np.int64)
        kernel_ =  np.reshape(kernels[i-1], (kernel_s, kernel_s, 1))
        cats_matrix_ = np.array([convulving_matrix(cats, kernel_, pad_method='same') for cats in cats_matrix])
        cats_in_disguise[:,:,:,i] = cats_matrix_[:, :, :]
        
    cats_in_disguise= np.vstack([cats_in_disguise[...,0],
                                 cats_in_disguise[...,1],
                                 cats_in_disguise[...,2],
                                 cats_in_disguise[...,3],
                                 cats_in_disguise[...,4],
                                 cats_in_disguise[...,5],
                                 cats_in_disguise[...,6],
                                 cats_in_disguise[...,7],
                                 cats_in_disguise[...,8],
                                 cats_in_disguise[...,9]])
    
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(10, 10, figsize=(60, 40),
                            sharex=True, sharey=True,
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0, wspace=0))

    for x, ax in enumerate(axes.flat):

        ax.imshow(cats_in_disguise[x, :, :], cmap='gray')
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.savefig("Cats_in_disguise.png")
    plt.show()
    

visualize_convolutions()