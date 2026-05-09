import torch
from matrices_vectors import scalar_matrix, matrix_sum

# --- Elementary Row Operations (EROs) ---

def row_interchange(R, i, j):# hw1
    """Interchanges row i and row j of matrix R."""
    # We use clone() to ensure we don't have reference issues during swapping
    temp = R[i].clone()
    # TODO: Perform row interchange
    # (Should done in HW1, not necessary to report)
    temp = R[i].clone()
    R[i] = R[j].clone()
    R[j] = temp
    return R

def row_scaling(R, i, s):# hw1
    """Scales row i of matrix R by a scalar s."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function to finish this row operation
    # Hint: To use the scalar_matrix, a row of 1D tensor should be reshaped to a row 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    # (Should done in HW1, not necessary to report)
    row_i_2d = R[i].reshape(1, cols)
    scaled_row = scalar_matrix(s, row_i_2d)
    R[i] = scaled_row.flatten()
    return R

def row_addition(R, i, j, s):# hw1
    """Adds s times row i to row j."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function and matrix_sum to finish this row operation
    # Hint: To use the scalar_matrix and matrix_sum, a row of 1D tensor should be reshaped to a row 
    # 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    # (Should done in HW1, not necessary to report)
    row_i_2d = R[i].reshape(1, cols)
    row_j_2d = R[j].reshape(1, cols)

    scaled_i = scalar_matrix(s, row_i_2d)
    new_row_j = matrix_sum(scaled_i, row_j_2d)
    R[j] = new_row_j.flatten()
    return R

# --- Gaussian Elimination using EROs ---

def gauss_elimination(A):# hw1
    """
    Transforms matrix A into RREF using the encapsulated row operations.
    """
    R = A.clone().to(torch.float32)
    rows, cols = R.shape
    
    pivot_row = 0
    pivot_col = 0

    # Set zero thresh (if entry is larger than this value, treat it as zero. Otherwise, treat it as nonzero)
    zero_thresh = 1e-6

    # TODO: Implement step 1 to step 4 in the lecture slides (the forward phase to Row Echelon Form) 
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions
    for j in range(cols):
        if pivot_row >= rows:
            break

        max_val, max_idx = torch.max(torch.abs(R[pivot_row:, j]), dim=0)
        max_idx = int(max_idx) + pivot_row

        if max_val < zero_thresh:
            continue

        R = row_interchange(R, pivot_row, max_idx)
        R = row_scaling(R, pivot_row, 1.0 / float(R[pivot_row, j]))

        for i in range(pivot_row + 1, rows):
            R = row_addition(R, pivot_row, i, -R[i, j])
        pivot_row += 1

    # TODO: Implement step 5 and step 6 in the lecture slides (the backward phase to Reduced Row Echelon Form) 
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions
    for i in range(rows - 1, -1, -1):
        # 找到第一個非零元素作為樞紐
        p_col = -1
        for j in range(cols):
            if abs(R[i, j]) > zero_thresh:
                p_col = j
                break
        if p_col != -1:
            R = row_scaling(R, i, 1.0/float(R[i,p_col]))

            for k in range(i - 1, -1, -1):
                R = row_addition(R, i, k, float(-R[k, p_col]))
                
    return R
