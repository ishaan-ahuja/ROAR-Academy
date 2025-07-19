# Please write a function set_array(L, rows, cols), where L is a list-type variable
# with rows*cols total number of elements. set_array returns a numpy array of
# rows-by-cols dimension.
# Note: The implementation can pick either row-order or column-order as the
# default order to fill the array with L elements. Alternatively, one can create a
# third input argument known as order.

import numpy as np

def set_array(L, rows, cols):
    """
    Create a numpy array from a list L with specified rows and columns.
    
    Parameters:
    L (list): The input list containing elements to fill the array.
    rows (int): The number of rows in the resulting array.
    cols (int): The number of columns in the resulting array.
    
    Returns:
    np.ndarray: A numpy array of shape (rows, cols) filled with elements from L.
    """
    if len(L) != rows * cols:
        raise ValueError("The length of L must be equal to rows * cols.")
    
    return np.array(L).reshape((rows, cols))  # Default is row-major order
# Example usage:
L = [1, 2, 3, 4, 5,
    6, 7, 8, 9]
rows = 3
cols = 3
array = set_array(L, rows, cols)
print(array)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
