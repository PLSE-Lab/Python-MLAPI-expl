import numpy as np

# create a n.n matrix
def createSquareMatrix(n):
    matrix = np.random.random((n, n))
    return matrix

# Calculate det() of a matrix
def determinant_recursive(matrix_numpy, total=0):
    # convert numpy array to list
    # np.array([[1, 2], [3, 4]]) => [[1, 2], [3, 4]]
    if type(matrix_numpy) is not list:
        matrix = matrix_numpy.tolist()

    # Store indices in list for row referencing
    indices = list(range(len(matrix)))
    # 2x2 matrix
    if len(matrix) == 2 and len(matrix[0]) == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return val

    # submatrix for focus column (fc)
    for fc in indices:  # for each focus column, ...
        # find the submatrix ...
        s_matrix = matrix.copy()  # make a copy
        s_matrix = s_matrix[1:]  # remove the first row
        height = len(s_matrix)

        for i in range(height):
            # for each remaining row of submatrix ...
            # remove the focus column elements
            s_matrix[i] = s_matrix[i][0:fc] + s_matrix[i][fc + 1:]

        sign = (-1) ** (fc % 2)
        # pass submatrix recursively
        sub_det = determinant_recursive(s_matrix)
        # total all returns from recursion
        total += sign * matrix[0][fc] * sub_det

    return total

# Calculate inverse()
def inversion_matrix(matrix_np):
    AM = matrix_np.tolist()
    n = len(AM)
    IM = np.eye(n)

    indices = list(range(n))  # to allow flexible row referencing
    for fd in range(n):  # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n):  # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler

            # SECOND: operate on all rows except fd row.
        for i in indices[0:fd] + indices[fd + 1:]:  # *** skip row with fd in it.
            crScaler = AM[i][fd]  # cr stands for "current row".
            for j in range(n):  # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]

    return np.multiply(matrix_np, IM)

# Main
matrix = createSquareMatrix(3)
print('Matrix:')
print(matrix)
print("==============")
print('inversion_matrix:')
print(inversion_matrix(matrix))
print("===============")
print('inversion_matrix_numpy:')
print(np.linalg.inv(matrix) )