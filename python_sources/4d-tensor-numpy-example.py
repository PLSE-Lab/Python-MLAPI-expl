'''
Aug12-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for example of a 4d tensor
results:
x.shape
x.ndim
(4, 2, 3, 4)
4
'''
import numpy as np

x  = np.array(

[
  [
              [ [1,2,3,4],
                [4,5,6,7],
                [7,6,3,2] ],
              [ [5,4,5,7],
                [4,5,7,4],
                [3,6,4,2] ]
      ],
  [
              [ [1,2,3,4],
                [4,5,6,7],
                [7,6,3,2] ],
              [ [5,4,5,7],
                [4,5,7,4],
                [3,6,4,2] ]
      ],
  [
              [ [1,2,3,4],
                [4,5,6,7],
                [7,6,3,2] ],
              [ [5,4,5,7],
                [4,5,7,4],
                [3,6,4,2] ]
      ],
  [
              [ [1,2,3,4],
                [4,5,6,7],
                [7,6,3,2] ],
              [ [5,4,5,7],
                [4,5,7,4],
                [3,6,4,2] ]
      ]
]
 )

print('x.shape')
print('x.ndim')

print(x.shape)
print(x.ndim)