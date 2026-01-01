# Song song hóa thuật toán Phân rã giá trị kỳ dị (SVD) và Ứng dụng trong Giảm chiều dữ liệu

## Giới thiệu
Đồ án này tập trung nghiên cứu và triển khai thuật toán **Phân rã giá trị kỳ dị (Singular Value Decomposition - SVD)** với trọng tâm là tối ưu hóa hiệu năng thông qua kỹ thuật tính toán song song trên hệ thống đa lõi (Multi-core CPU). Mục tiêu là giải quyết bài toán giảm chiều dữ liệu cho các tập dữ liệu lớn, nơi các phương pháp tuần tự truyền thống gặp giới hạn về thời gian xử lý.

## Các thuật toán đã triển khai
Dự án bao gồm việc xây dựng từ đầu (from scratch) các thuật toán song song sau:

1.  **Nhân ma trận song song:**
    *   **Thuật toán Cannon:** Sử dụng kỹ thuật chia khối (blocking) và dịch chuyển dữ liệu theo lưới 2D.
    *   **Thuật toán DNS (Dekel-Nassimi-Sahni):** Phân chia dữ liệu theo khối 3D.
2.  **SVD song song:**
    *   Sử dụng biến thể **Parallel Norm-Reducing Jacobi** để tính toán giá trị riêng và vector riêng cho ma trận hiệp phương sai.
    *   Chiến lược chia cặp và xoay vòng (Round-Robin) để khử song song các phần tử ngoài đường chéo.
3.  **Các thuật toán hỗ trợ:**
    *   Chuyển vị ma trận song song (Blocked Parallel Transpose).
    *   Sắp xếp song song (Parallel Sort) để sắp xếp giá trị kỳ dị.

## Kết quả thực nghiệm
Thử nghiệm được thực hiện trên vi xử lý **AMD Ryzen 5 6600HS** (6 nhân, 12 luồng).

*   **Nhân ma trận (Cannon):** Đạt hệ số tăng tốc (speedup) lên tới **11.3 lần** trên 12 luồng (vượt lý thuyết nhờ tối ưu hóa Cache L1/L2).
*   **SVD song song:** Đạt hệ số tăng tốc **4.6 lần** với các ma trận kích thước lớn ($N > 3000$).
*   **Ứng dụng:** Áp dụng thành công vào bài toán giảm chiều dữ liệu ảnh **MNIST** (từ 784 chiều xuống 50 chiều), duy trì độ chính xác phân loại $\approx 97\%$ với mô hình MLP.

## Mã nguồn
Mã nguồn đầy đủ của dự án được lưu trữ tại:
👉 **[GitHub Repository](https://github.com/minhhhduc/svd)**

## Cấu trúc thư mục
*   `source/`: Mã nguồn C/C++ của các thuật toán.
*   `include/`: Các file header.
*   `demo/`: Jupyter Notebook minh họa ứng dụng trên MNIST.
*   `LATEX_template/`: Báo cáo chi tiết dạng LaTeX.

---
*Đồ án môn học Tính toán song song.*
