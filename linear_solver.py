import torch
from matrices_vectors import matrix_vector_product
from gauss import gauss_elimination
from determinants import calculate_determinant

#向量b是否在矩陣M的 Column Space 裡面，等於問向量b是否在矩陣M各個行的 Span 裡面
def in_column_space(matrix_M, vector_b):#TD done
    """
    Checks if vector_b lies within the column space of matrix_M.
    """
    # TODO: Replace the following line
    # Hint: use test_span.
    if test_span(matrix_M, vector_b):
        return True
    else:
        return False

def in_null_space(matrix_M, vector_b):#TD done
    """
    Checks if vector_b is in the null space of matrix_M.
    """
    zero_threshold = 1e-6
    
    # TODO: Replace the following line
    # Hint: Use matrix_vector_product
    # Note: When the absolute value is less than zero_threshold, we treat it as zero.
    if torch.all(torch.abs(matrix_vector_product(matrix_M, vector_b))<zero_threshold):
        return True
    else:
        return False
    

def test_invertibility_by_determinants(matrix_M):#TD done
    """
    Checks if a square matrix is invertible using its determinant.
    Logic: A matrix is invertible if and only if its determinant is non-zero.
    """
    zero_threshold = 1e-6

    rows, cols = matrix_M.shape

    # 1. Invertibility is only defined for square matrices
    if rows != cols:
        return False

    # TODO: Replace the following line
    # Hint: Use calculuate_determinants
    # Note: When the absolute value is larger than zero_threshold, we treat it as nonzero.
    if abs(calculate_determinant(matrix_M)) < zero_threshold:
        return False
    else:
        return True


def test_invertibility(matrix_M):# hw3
    """
    Checks if a square matrix is invertible.
    """
    rows, cols = matrix_M.shape

    # 1. Invertibility is only defined for square matrices
    if rows != cols:
        return False

    # 2. TODO: Impelement the invertibility check by using the property:
    # A square matrix is invertible if and only if its Reduced Row Echelon Form (RREF)
    # is the Identity Matrix.
    # (Should done in HW3, not report)
    rref_M = gauss_elimination(matrix_M)

    I = torch.eye(rows, dtype=torch.float32)
    error = torch.norm(rref_M - I).item()
    if error < 1e-5:
        return True
    else:
        return False

def test_span(matrix_M, vector_b):# hw2
    """
    TODO: Determine if vector_b is in the span of the columns of matrix_M.
    Hint: Does Mx = b have a solution?
    Hint: Use solve_linear_equations
    # (Should done in HW2, not report)
    """
    solution = solve_linear_equations(matrix_M, vector_b)
    return solution is not None

def test_linear_dependence(matrix_M):# hw2
    """
    TODO: Determine if the columns of matrix_M are linearly dependent.
    
    Logic: The columns are linearly dependent if there exists a NON-ZERO 
           vector 'x' such that Mx = 0.
    
    Challenge: Our 'solve_linear_equations' might return the trivial x=0.
    Hint: If the system has a unique solution (only x=0), they are Independent.
          If the system has infinite solutions (free variables), they are Dependent.
          Think about how your solver handles singular matrices.
    Hint: Nonzero: Euclidean norm > small epsilon (ex: 1e-6)
    Hint: Use solve_linear_equations
    # (Should done in HW2, not report)
    """
    rows, cols = matrix_M.shape
    # 建立零向量 b
    zero_b = torch.zeros((rows, 1), dtype=torch.float32)
    
    x = solve_linear_equations(matrix_M, zero_b)
    
    epsilon = 1e-6
    
    # 計算向量長度: ||x||
    norm_x = torch.norm(x)
    
    # 如果長度大於 epsilon，代表找到了一個非零組合能拼出零向量 -> 線性相依
    return norm_x > epsilon

def test_consistency(augmented_RREF):# hw1
    """
    Checks if the system Ax = b is consistent.
    """
    rows, cols = augmented_RREF.shape
    # TODO: Use the following property to determine consistency:
    # A system is inconsistent if and only if a row in the RREF looks like [0 0 ... 0 | b_i] 
    # where b_i != 0. 
    # Return False if inconsistent 
    # (Should done in HW1, not report)
    zero_thresh = 1e-6
    for i in range(rows):
        # 檢查係數部分是否全為 0
        if torch.all(torch.abs(augmented_RREF[i, :-1]) < zero_thresh):
            # 如果係數為 0 但增廣常數項 b 不為 0 -> 不一致
            if abs(augmented_RREF[i, -1]) > zero_thresh:
                return False
            
    return True # Consistent

def generate_column_space_basis(matrix_M):#TD done
    """
    Extracts the basis of the Column Space of matrix_M.
    Logic: 
    1. Perform Gaussian Elimination to get RREF.
    2. Identify the pivot columns (columns with leading 1s).
    3. The corresponding columns in the ORIGINAL matrix form the basis.
    """
    rows, cols = matrix_M.shape
    
    # 1. Get the RREF of the matrix
    rref = gauss_elimination(matrix_M)
    
    pivot_column_indices = []
    
    # 2. TODO: Identify pivot columns
    # We look for the first non-zero entry in each row of the RREF
    zero_thresh = 1e-6
    for i in range(rows):
        for j in range(cols):
            if abs(rref[i,j]) > zero_thresh:
                pivot_column_indices.append(j)
                break 
    
    # If no pivots found (zero matrix), return an empty tensor or handled case
    if not pivot_column_indices:
        return torch.empty((rows, 0))
        
    # 3. Extract those columns from the ORIGINAL matrix M
    basis_matrix = matrix_M[:, pivot_column_indices]
    
    return basis_matrix

def generate_null_space_basis(matrix_M):#TD done
    """
    Generates a basis for the null space of matrix_M.
    Logic: 
    1. Transform matrix M to Reduced Row Echelon Form (rref).
    2. Identify pivot and free variable columns.
    3. Express basic variables in terms of free variables to form basis vectors.
    """
    # 1. Get the Reduced Row Echelon Form
    rref = gauss_elimination(matrix_M)
    rows, cols = rref.shape
    
    # 2. TODO: Identify pivot columns (where leading 1s are located)
    pivot_cols = {}
    zero_thresh = 1e-6
    for i in range(rows):
        for j in range(cols):
            if abs(rref[i,j]) > zero_thresh:
                pivot_cols[j] = i
                break 
            
    # 3. TODO: Identify free variable columns
    free_cols = []
    for j in range(cols):
        if j not in pivot_cols:
            free_cols.append(j)
    
    # If no free variables exist, the null space contains only the zero vector
    if not free_cols:
        return torch.zeros((cols, 1))
    
    # 4. TODO: Construct one basis vector for each free variable, append the vector into basis_vectors
    basis_vectors = []

    for f in free_cols:
        #建立一個全為 0 的直向量，長度等於矩陣行數 (變數)
        v = torch.zeros((cols, 1), dtype=torch.float32)
        
        #將自由變數設為 1
        v[f, 0] = 1.0
        
        #利用 RREF 回推基本變數的值
        for p_col, p_row in pivot_cols.items():
            # 將方程式移項：基本變數 = - (係數 * 自由變數)
            v[p_col, 0] = -float(rref[p_row, f])
            
        #將構造好的基底向量加入
        basis_vectors.append(v)
        
    # Combine all basis vectors into a single basis matrix
    return torch.cat(basis_vectors, dim=1)

def generate_solution(augmented_RREF):# hw1
    """
    Extracts a solution vector. Basic variables are solved, Free variables are set to 1.
    """
    rows, cols_aug = augmented_RREF.shape
    num_vars = cols_aug - 1
    solution = torch.ones((num_vars, 1), dtype=torch.float32) # Default free variables to 1

    # TODO: Identify pivot columns
    # Hint: Find the first nonzero in each row
    # (Should done in HW1, not necessary to report)
    
    # TODO: Calculate basic variables
    # Hint: Note that we have assume all free variables are set to 1
    # (Should done in HW1, not necessary to report)

    zero_thresh = 1e-6

    pivot_cols = {}
    for i in range(rows):
        for j in range(num_vars):
            if abs(augmented_RREF[i, j] - 1.0) < zero_thresh:
                pivot_cols[j] = i  # 記錄變數 j 是由哪一列決定的
                break

    for j in range(num_vars):
        if j in pivot_cols:
            row_idx = pivot_cols[j]
            # x_j = b_i - sum(A_ik * x_k) 對於所有自由變數 k
            val = augmented_RREF[row_idx, -1]
            for k in range(j + 1, num_vars):
                val -= augmented_RREF[row_idx, k] * solution[k, 0]
            solution[j, 0] = val
            
    return solution


def solve_linear_equations_by_inverse(A, b):# hw3
    """
    Solves Ax = b using the inverse matrix method: x = A^(-1)b.
    The inverse is calculated via Gaussian elimination on [A | I].
    """
    # 1. First, check if the matrix is invertible
    if not test_invertibility(A):
        return None

    num_vars = A.shape[1]
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    #TODO: Implement the matrix inversion method to get the inverse of A 
    # (Should done in HW3, not report)
    #TODO: Solve the solution by using inverse of A and return the result
    # (Should done in HW3, not report)
    I = torch.eye(num_vars, dtype=torch.float32)
    augmented_A_I = torch.cat((A, I), dim=1)
    rref_matrix = gauss_elimination(augmented_A_I)
    A_inverse = rref_matrix[:, num_vars:]

    solution = matrix_vector_product(A_inverse, b)

    return solution


def solve_linear_equations(A, b):# hw1
    """
    Wrapper function to solve Ax = b.
    """

    # Create Augmented Matrix [A | b]
    augmented_matrix = torch.cat((A, b), dim=1)

    rows, cols_aug = augmented_matrix.shape
    num_vars = cols_aug - 1
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    
    # TODO: Use our custom gauss_elimination to obtain the Reduced Row Echelon Form.
    # Then, use test_consistency to inspect the reduced matrix.
    # Return None if the system is inconsistent. Otherwise, return the results of generate_solution.
    # (Should done in HW1, not report)
    rref_matrix = gauss_elimination(augmented_matrix)
    
    if not test_consistency(rref_matrix):
        print("System is inconsistent!")
        return None
    
    solution = generate_solution(rref_matrix)
    
    return solution
