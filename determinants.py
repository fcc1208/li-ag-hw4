import torch
from gauss import row_interchange, row_addition

def calculate_determinant(matrix_M):#TD done
    """
    Calculates the determinant of a square matrix using the Gaussian elimination 
    logic (Forward Phase only) defined in gauss.py.
    
    Logic: 
    1. Transform the matrix to Row Echelon Form (REF).
    2. Tracking row swaps to flip the sign.
    3. Determinant = ((-1)^swaps) * product of diagonal entries.
    """
    R = matrix_M.clone().to(torch.float32)
    rows, cols = R.shape
    
    if rows != cols:
        return 0.0

    sign = 1.0
    pivot_row = 0
    pivot_col = 0

    # --- TODO: Forward Phase (Transforming to REF) ---
    # Hint: Use row interchange and row addition, not use row scaling
    # Hint: Implement the following while loop, update pivot_row and pivot_col properly
    # while pivot_row < rows and pivot_col < cols:
    #     ......
    zero_thresh = 1e-6
    while pivot_row < rows and pivot_col < cols:
        #尋找當前 column 中，絕對值最大的元素作為 Pivot
        max_val, max_idx = torch.max(torch.abs(R[pivot_row:,pivot_col]), dim=0)
        max_idx = int(max_idx) + pivot_row

        #如果最大值極小，代表這個 column 下方全是 0，跳過並換下一個 column
        if max_val < zero_thresh:
            pivot_col +=1
            continue

        #如果 Pivot 不在該列，進行列交換，並且將行列式符號乘上 -1
        if max_idx != pivot_row:
            R = row_interchange(R, pivot_row, max_idx)
            sign *= -1.0

        pivot_val = float(R[pivot_row, pivot_col])

        for i in range(pivot_row + 1, rows):
            factor = float(R[i, pivot_col])/pivot_val
            if abs(factor) > zero_thresh:
                R = row_addition(R, pivot_row, i, -factor)

        #處理完 Pivot 後，往右下方移動
        pivot_row +=1
        pivot_col +=1

    # --- Final Calculation ---
    # TODO: Replace the following line
    # Hint: The determinant is the sign-adjusted product of the diagonal elements
    det = sign

    #經過Forward Phase之後，矩陣R已經變成了一個上三角矩。對於三角矩來說，它的Eigenvalue或是它的行列式，
    #會等於主對角線上所有元素的乘積
    for i in range(rows):
        det *= float(R[i, i])
    
    return det
