import torch
import numpy as np

def scalar_matrix(s, M):# hw1
    rows = M.shape[0]
    cols = M.shape[1]
    # Initialize a tensor of zeros with the same shape as M
    sM = torch.zeros(M.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to scale each entry (should done in HW1)
    # (Should done in HW1, NOT Necessary to report)
    for i in range(rows):
        for j in range(cols):
            sM[i,j] = s * M[i,j]
    return sM

def matrix_sum(M1, M2):# hw1
    rows = M1.shape[0]
    cols = M1.shape[1]
    # Initialize a tensor of zeros with the same shape as M1
    M = torch.zeros(M1.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to add 
    # corresponding entries of M1 and M2 in a general manner. (should done in HW1)
    # (Should done in HW1, NOT Necessary to report)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = M1[i, j] + M2[i, j]
    return M

def matrix_vector_product(M, vec):# hw1
    rows, cols = M.shape[0], M.shape[1]
    vec2 = torch.zeros((rows, 1), dtype=torch.float32)
    
    # TODO: Complete the functionality by implementing a for loop for a general linear combination.
    # Hint: Utilize the scalar_matrix() and matrix_sum() functions. (should done in HW1)
    # (Should done in HW1, NOT Necessary to report)
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += M[i, j] * vec[j, 0]
        vec2[i, 0] = row_sum
    return vec2

def matrix_multiplication(M1, M2):# hw1
    m, n = M1.shape[0], M1.shape[1] # M1 is an m x n matrix
    _, p = M2.shape[0], M2.shape[1] # M2 is an n x p matrix
    M3 = torch.zeros((m, p), dtype=torch.float32)
    
    # TODO: Complete the functionality by implementing a for loop for matrix multiplication.
    # Hint: Utilize the matrix_vector_product for the implementation.
    # Hint: You may need reshape and flatten to control the shapes of vectors.
    for j in range (p):
        vec_j = M2[:,j].reshape(n,1)# M2 j行單獨抽出來 reshape成 (n * 1)2D
        result_vec = matrix_vector_product(M1, vec_j)
        M3[:,j] = result_vec.flatten()# 2D 壓回 1D
    return M3

def compute_rotation_matrix_2d(theta):# hw2
    # Initialize a 2x2 identity matrix 
    rot = torch.eye(2, dtype=torch.float32)
    # TODO: Complete the functionality by specifying a 2D rotation matrix
    # Using torch.cos and torch.sin for tensor-compatible math
    # theta > 0: counter-clockwise
    c = np.cos(theta)
    s = np.sin(theta)
    #torch.cos會報錯(規定輸入必是一個 Tensor，但傳進去的是一個的 float)，改np.

    rot[0,0] = c
    rot[0,1] = -s
    rot[1,0] = s
    rot[1,1] = c

    return rot

def compute_y_mirror_matrix_2d():# hw2
    # Initialize a 2x2 identity matrix 
    mirror = torch.eye(2, dtype=torch.float32)
    # TODO: Complete the functionality by specifying a y-axis mirror matrix
    # This matrix flips the x-coordinate: [[-1, 0], [0, 1]]
    
    mirror[0,0] = -1
    mirror[0,1] = 0
    mirror[1,0] = 0
    mirror[1,1] = 1

    return mirror