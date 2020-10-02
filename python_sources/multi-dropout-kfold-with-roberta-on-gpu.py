#!/usr/bin/env python
# coding: utf-8

# ## Multi-Sample Dropout
# Dropout is an efficient regularization instrument for avoiding overfitting of deep neural networks. It works very simply randomly discarding a portion of neurons during training; as a result, a generalization occurs because in this way neurons depend no more on each other.
# In this post, I try to reproduce the results presented in this paper; which introduced a technique called Multi-Sample Dropout. As declared by the author, its scopes are:
# * accelerate training and improve generalization over the original dropout;
# * reduce computational cost because most of the computation time is consumed in the layers below (often convolutional or recurrent) and the weights in the layers at the top are shared;
# * achieve lower error rates and losses.
# ![](https://miro.medium.com/max/700/1*mdBXp3-D7G7mTcKDZHui8g.png)

# ## For Keras

# Dropout parameters change 0.1 - 0.5. You can customer this for betters model.

# In[ ]:


def get_model(cfg):
    model_input = tf.keras.Input(shape=(cfg['net_size'], cfg['net_size'], 3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)
    
    outputs = []    
    for i in range(cfg['net_count']):
        constructor = getattr(efn, f'EfficientNetB{i}')
        
        x = constructor(include_top=False, weights='imagenet', 
                        input_shape=(cfg['net_size'], cfg['net_size'], 3), 
                        pooling='avg')(dummy)
        dense = []
        FC = tf.keras.layers.Dense(32, activation='relu')
        for p in np.linspace(0.1,0.5, 5):
            x_ = tf.keras.layers.Dropout(p)(x)
            x_ = FC(x_)
            x_ = tf.keras.layers.Dense(1, activation='sigmoid')(x_)
            dense.append(x_)
        x = tf.keras.layers.Average()(dense)
        outputs.append(x)
        
    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()
    return model


# ## For pytorch

# In[ ]:


class multilabel_dropout():
    # Multisample Dropout: https://arxiv.org/abs/1905.09788
    def __init__(self, HIGH_DROPOUT, HIDDEN_SIZE):
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(config.HIDDEN_SIZE * 2, 2)
    def forward(self, out):
        return torch.mean(torch.stack([
            self.classifier(self.high_dropout(p))
            for p in np.linspace(0.1,0.5, 5)
        ], dim=0), dim=0)

