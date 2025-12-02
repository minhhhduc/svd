import numpy as np


# Đọc dữ liệu từ file, tách từng số trên mỗi dòng
data = []
with open('d:/C++/svd/svd/data/input/matrix_20.txt', 'r') as f:
    for line in f:
        for val in line.strip().split():
            try:
                data.append(float(val))
            except ValueError:
                continue

# Lấy kích thước ma trận từ 2 số đầu tiên
if len(data) < 2:
    print("Không đủ dữ liệu để xác định kích thước ma trận.")
    exit()
nrow, ncol = int(data[0]), int(data[1])
data = data[2:]


# Chỉ lấy đúng nrow x ncol số cho ma trận A
print(f"Kích thước ma trận từ file: {nrow}x{ncol}")
if len(data) < nrow * ncol:
    print(f"Không đủ dữ liệu để tạo ma trận {nrow}x{ncol}.")
    exit()
A = np.array(data[:nrow*ncol]).reshape((nrow, ncol))
print("Ma trận đầu vào (một phần):")
print(A[:min(5, nrow), :min(5, ncol)])



# Tính eigenvalues của C = A^T A (178x178) để so sánh với Jacobi
if ncol <= nrow:
    C = A.T @ A
    print(f"Tính eigenvalues của C = A^T A, kích thước {ncol}x{ncol}")
    eigvals = np.linalg.eigvals(C)
    eigvals = np.sort(eigvals)[::-1]
    print("Eigenvalues (numpy):")
    for i, val in enumerate(eigvals):
        print(f"w[{i}] = {val.real:.10f}{val.imag:+.10f}i")
else:
    print("Số cột lớn hơn số hàng, không thể tính A^T A đúng.")
