# Revision R1 — CMC ID 92446: hướng dẫn chạy phân tích và đọc kết quả

Thư mục này chứa bộ phân tích cho bản sửa vòng 1. Bộ phân tích **chỉ đọc dữ liệu đã có** trong `journal_cmc/results/`:
- **không** cần mở simulator Unity;
- **không** chạy lại thí nghiệm nào;
- **không** sửa file cũ nào.

Mọi kết quả được ghi vào `revision_R1/output/`.

## 1. Cách chạy (1 lệnh, khoảng 15 giây)

Mở PowerShell hoặc Terminal tại thư mục `SMC_Hough_CMC`, rồi chạy:

```
python revision_R1/run_all.py
```

Console đúng sẽ kết thúc như sau:

```
[run_all] 6/6 steps OK
[run_all] report : ...\revision_R1\output\REPORT.md
```

- Nếu thấy `x/6 steps OK` với x < 6, mở `revision_R1/output/run_log.txt` và kiểm tra thông báo lỗi.
- Cảnh báo `UserWarning: ... tight_layout` là vô hại, bỏ qua.

**Yêu cầu:** Python 3.10 cùng các gói numpy, scipy, matplotlib (máy hiện tại đã có đủ). Nếu thiếu, chạy:

```
pip install numpy scipy matplotlib
```

Muốn chạy riêng một phần (ví dụ để kiểm tra), dùng:

```
python revision_R1/analysis/s3_nominal.py
```

## 2. Kết quả nằm ở đâu

| File | Nội dung |
|---|---|
| `output/REPORT.md` | **Đọc file này trước**: toàn bộ con số, chia theo mục S1–S6 |
| `output/run_log.txt` | Bước nào OK hay FAIL |
| `output/tables/*.csv` | Bảng số liệu, mở bằng Excel |
| `output/tables/*.tex` | Các dòng LaTeX để dán vào bảng trong bài |
| `output/figures/*.png` | Hình đã vẽ lại, 400 dpi |

## 3. Mục nào trong REPORT.md trả lời ý nào của reviewer

| Mục | Nội dung | Dùng cho |
|---|---|---|
| S1 | Nghiên cứu optimizer cùng budget: mean ± SD, 95% CI, median, seed tệ nhất, Wilcoxon + Mann–Whitney + Holm, Cliff's δ, kiểm định độ phân tán, hội tụ, thời gian chạy | Bảng 2, Hình 2 · Editor 3 · R1-3 · R2-2, R2-3 (thời gian), R2-4 |
| S2.1 | Độ hợp lệ các lượt tune (PSO chỉ có 13/80 lượt hợp lệ) | R2-2 · thư gửi Editor |
| S2.2–2.3 | Mức của grid, gain triển khai, gain Tuned PID | Bảng 3 · Editor 3 · R2-2 |
| S2.4 | Số liệu luật điều khiển: K_P = 0.1763, K_D = 0.00625, 0.13% bước ra ngoài lớp biên, 1.31% tỉ lệ switching, giới hạn φ→∞ | R2-1 |
| S2.4 (cuối) | Tần số vòng lặp, tỉ lệ frame lặp | Editor 2 |
| S2.5 | Giá trị rᵢ (reference scales) | Editor 2 · R2-5 |
| S2.6 | η_θ, φ_θ qua 13 lượt BO | R2-5 · §3.3 |
| S2.7 | Phân tích độ nhạy trọng số/clip offline | R2-5 |
| S3 | Bảng 4 mới, 25 test Holm, respawn, bất định Pareto | Bảng 4 · Editor 4, 5, 6 · R1-1 · R2-4 |
| S4 | Bảng 5: run bị loại, 3 cách loại, tần số vòng theo điều kiện, 20 test Holm | Bảng 5 · Editor 4 · R1-2 |
| S5 | Bảng 6: n, SD, violation rate, 16 test Mann–Whitney + Holm | Bảng 6 · Editor 4, 6 |
| S6 | Hình 3 vẽ lại theo đúng định nghĩa SSI | Editor 6 |

**Cách đọc bảng test:**
- `p` là p-value gốc; `Holm p` là p-value sau hiệu chỉnh đa so sánh.
- `Sig. = yes` nghĩa là còn có ý nghĩa sau Holm (< 0.05). **Chỉ những dòng `yes` mới được viết là "significant" trong bài.**
- `Mean diff [95% CI]` = proposed trừ comparator. Số âm nghĩa là proposed thấp hơn, tức tốt hơn. Nếu khoảng CI chứa 0 thì không kết luận được khác biệt.

## 4. Những việc chỉ bạn làm được (không phải chạy code)

1. **Hình 1:** mở `paper_cmc/architecture_CMC.drawio` bằng draw.io.
   - Thêm nhánh tốc độ riêng: v từ simulator → e_v = v_ref − v → SMC tốc độ → u_v. Tách mũi tên "vehicle speed" ra khỏi khối Hough.
   - Đổi "GP surrogate + EI" thành "GP surrogate + log-EI".
   - Thay ảnh camera nhỏ bằng một khung hình khác Hình 1 của bài hội nghị, ví dụ `journal_cmc/results/perception_debug/sample1_raw.png`.
   - Xuất file: File → Export as → PDF, hoặc PNG ở mức ≥ 600 dpi.
2. **Thông tin môi trường Unity mà bản build không cho biết:** chiều dài vòng đua, bán kính cong, bề rộng làn (m), FOV và vị trí lắp camera, thông số xe, bản build có được phép chia sẻ không. Không có thông tin nào thì bài sẽ ghi rõ là "không truy xuất được từ bản build".
3. **Quyết định cách xử lý PSO-SMC** (lượt tune bị hỏng):
   - (a) bỏ PSO-SMC khỏi Bảng 3, Bảng 4 và Hình 3, và nêu lý do (khuyến nghị); hoặc
   - (b) giữ lại kèm ghi chú.
4. **Chia sẻ code:** có tạo repo GitHub công khai không, hay giữ "available upon reasonable request"?
