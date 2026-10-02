# QUICKSTART — chạy thực nghiệm bằng tay

> Thẻ hướng dẫn ngắn. Chi tiết + output dự kiến từng lệnh: xem **README_RUN.md**.
> Mọi lệnh chạy trong thư mục này (`journal_cmc/`).

---

## ✅ Chuẩn bị 1 lần — tạo môi trường (xem **SETUP_ENV.md** để biết chi tiết)

```powershell
cd journal_cmc   # from the repository root
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1   # tạo env 'smc_cmc' + cài đủ (1 lần)
```

## ✅ Mỗi lần chạy — kích hoạt env rồi kiểm tra

```powershell
conda activate smc_cmc
python run.py gpucheck
```
Phải thấy: `CUDA available: True`, `NVIDIA GeForce RTX 3060`, `BO backend that will be used: botorch`.

> ⚠️ **Đừng chạy trong conda `(base)`** — base của bạn là torch CPU. Luôn `conda activate smc_cmc`.

---

## ✅ Trước MỖI buổi chạy: mở map

Nhấn đúp **`Map_demo_v3/Window/Map_demo_v3.exe`**. Để cửa sổ Unity chạy suốt buổi
(mở **1 lần** cho cả loạt lệnh). Đừng mở `mock_server.py` cùng lúc (trùng cổng 54321).

---

## ✅ Bước 0 — thử kết nối (30 giây)

```bash
python run.py demo --method ManualSMC --show
```
Hiện cửa sổ vẽ lane + lệnh lái, in bảng metric. Nếu chạy được → sẵn sàng. Nhấn `q` để dừng.

---

## ✅ 3 mức chạy (profile) — quy trình khuyến nghị

| Lệnh | Thời gian | Dùng khi |
|---|---|---|
| `python run.py --quick all` | ~25 phút | Smoke-test flow. Số liệu KHÔNG dùng cho paper. |
| `python run.py --profile medium all` | ~2.5–3.5h | **Kiểm chứng gần-cuối** (ít noise) trước khi chạy full. |
| `python run.py --profile full all` | ~11h | Số liệu chính thức cho paper. |

Đi tuần tự: `--quick` → `--profile medium` → `--profile full`. Mỗi lần lưu JSON riêng;
`python run.py analyze` dựng lại bảng/hình offline bất cứ lúc nào.

---

## ✅ Cách 2 — chạy THẬT cho paper (chạy từng bước, ~11 giờ tổng, có thể rải nhiều buổi)

Chạy theo đúng thứ tự (mỗi lệnh lưu JSON riêng, dừng giữa chừng vẫn giữ được dữ liệu):

```bash
python run.py reference        # [1] ~3'   chuẩn hoá objective (Manual-SMC × 3 episode)
python run.py optimize --which all   # [2] ~3.8h  tune SMC (random/grid/pso/bo) + tune PID(bo)
python run.py compare          # [3] ~1.3h  6 method × 15 seed        → Table 3, Fig 3/5
python run.py robustness       # [4] ~2.1h  nhiễu thị giác × method   → Table 4, Fig 4
python run.py ablation         # [5] ~4h    5 biến thể SMC            → Table 5
python run.py analyze          # [6] ~10''  dựng 7 bảng + 5 hình (offline, KHÔNG cần map)
```

> **Bắt buộc:** `reference` → `optimize` phải xong trước `compare/robustness/ablation`
> (chúng cần best gains + hệ số chuẩn hoá). `analyze` chạy được bất cứ lúc nào, dựng bảng
> nào đã có dữ liệu.

**Rải nhiều buổi:** cứ chạy vài lệnh rồi tắt; buổi sau mở lại map, chạy tiếp lệnh còn lại.
Cuối cùng `python run.py analyze` gom tất cả. Muốn giữ nhiều lần chạy → copy cả thư mục
`results/` ra chỗ khác trước khi chạy lại.

**✅ Checkpoint/Resume (chống map đơ giữa chừng):** `compare`, `robustness`, `ablation`
tự lưu sau MỖI episode/biến thể. Nếu map chết giữa chừng, **chỉ cần chạy LẠI đúng lệnh đó**
— nó tự bỏ qua phần đã xong, chạy tiếp phần còn thiếu (ablation còn tái dùng gains BO đã
tune, khỏi tối ưu lại). Muốn tính lại từ đầu (vd sau khi đổi tham số): thêm `--fresh`.
Guard cũng tự ép reconnect để thử reset xe kẹt trước khi bỏ cuộc.

**Muốn nhanh hơn** (giảm giờ máy): mở `config.py`, giảm `T_MEASURE_FULL` (45→30),
`BUDGETS['full']['bo_n_calls']` (60→40), `compare_seeds/robustness_seeds` (→8–10).
**Muốn chắc hơn cho bài cuối:** đặt `OPT_EVAL_SEEDS = 2` (J bớt nhiễu, chậm gấp đôi).

---

## ✅ REVIEWER — so sánh optimizer CÔNG BẰNG (bắt buộc cho revision)

Reviewer chê Table 6 cũ so sánh optimizer ở **budget mặc định khác nhau** (BO 60, PSO 80,
Random 40, Grid 27) và Grid chỉ tìm 3/6 gain → không công bằng. Lệnh mới sửa đúng điều đó:
**cùng không gian 6-D, cùng số eval, ≥10 seed** cho random/pso/bo.

```bash
python run.py faircompare                       # full: 60 eval x 10 seed x {random,pso,bo}
python run.py --profile medium faircompare      # episode ngắn hơn (20s) → nhanh hơn
python run.py faircompare --evals 40 --seeds 8  # rẻ hơn nhưng vẫn công bằng
python run.py faircompare --with-grid           # chạy thêm grid 3-D (tham chiếu, không seed)
```

> ⚠️ **Thứ tự tham số:** cờ chung (`--profile`) đặt **TRƯỚC** `faircompare`; cờ riêng
> (`--evals/--seeds/--with-grid`) đặt **SAU**. Ví dụ: `python run.py --profile medium faircompare --evals 60 --seeds 10`.

**Thời gian:** mỗi eval là 1 episode thời-gian-thực (~51s ở full). 60 eval ≈ 51 phút/lần;
cả lưới {random,pso,bo} × 10 seed ≈ **~25h**. Dùng `--profile medium` (24s/eval → ~12h) hoặc
giảm `--evals/--seeds` để rút ngắn. **Có checkpoint**: map đơ giữa chừng chỉ cần chạy LẠI đúng
lệnh đó, nó bỏ qua các (optimizer, seed) đã xong.

**Kết quả** (in ra cuối + lưu file):
- `results/optim/fair_compare_results.json` — dữ liệu thô mọi lần chạy
- `results/optim/fair_compare_summary.json` — best-J mean±SD + Wilcoxon BO-vs-PSO/Random
- `results/tables/fair_convergence.csv` — đường hội tụ mean±SD/eval (để vẽ lại Fig 2)
- Cuối màn hình in sẵn **khối PASTE-READY** để dán thẳng vào Table 6 của bài.

**Cập nhật bài sau khi chạy xong:** thay các dòng số trong bảng *Optimization Efficiency*
(label `tab:efficiency`) bằng khối PASTE-READY (cột "Best objective" → `mean±SD`, cùng budget
cho mọi optimizer), và vẽ lại Fig `fig:convergence` từ `fair_convergence.csv`.

---

## ✅ Lấy kết quả ở đâu

```
results/tables/   table2..table7  (.csv để xem, .tex để dán vào paper)
results/figures/  fig2..fig5 .png
results/optim/    *.json  (dữ liệu gốc, tái dựng bảng/hình bằng `analyze`)
results/logs/     quỹ đạo thô từng episode
```

| Bảng | Trả lời | Điểm nhấn |
|---|---|---|
| Table 3 | RQ1/RQ2 tổng hợp | có cả *PID (untuned)* và *Tuned PID (BO)* |
| Table 4 | RQ3 robustness | nhiễu clean/noise/shadow/occlusion/blur |
| Table 5 | novelty ablation | ±boundary layer, ±Lyapunov |
| Table 6 | hiệu quả optimizer | BO(GPU) vs PSO/Grid/Random |
| **Table 7** | ý nghĩa thống kê | **`BO_SMC vs TunedPID`** = SMC thắng *dù cả hai đều tune* |

---

## ⚠️ 4 điều nhớ khi VIẾT paper (đọc README §10 & §10b)

1. Lệch làn đo bằng **pixel** (+ %lane-width xấp xỉ) — khai báo Limitation, đừng ghi mét.
2. Lyapunov là **"inspired penalty"**, không phải chứng minh ổn định.
3. Robustness từ **nhiễu tiêm trong Python** (map chỉ 1 track), không phải nhiều bản đồ.
4. So sánh công bằng: dùng **Tuned PID** + **ablation** để tách "cấu trúc SMC" khỏi "tuning".

---

## 🔧 Sự cố thường gặp

| Lỗi | Xử lý |
|---|---|
| `Cannot connect ...:54321` | chưa mở `Map_demo_v3.exe` |
| `No optimizer result at ...` khi compare | chạy `optimize` trước |
| BO in `device=cpu` | thiếu botorch/CUDA → `pip install botorch`, `run.py gpucheck` |
| Muốn test không có map | terminal khác: `python mock_server.py` rồi chạy lệnh (số liệu chỉ để test) |
