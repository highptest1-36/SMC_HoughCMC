# journal_cmc — Hướng dẫn chạy đầy đủ

Framework tối ưu hoá tham số **Boundary-Layer Sliding Mode Control** bằng
**Bayesian Optimization (chạy trên GPU RTX 3060)** cho bài toán giữ làn, phục vụ
bản journal CMC (EIDT2026) mở rộng từ bài conference (Springer 978-981-95-6111-7_19).

> Hướng novelty: *"Lyapunov-Penalized Bayesian Optimization of Boundary-Layer
> Sliding Mode Control for Real-Time Lane Keeping"*. Novelty nằm ở **lớp điều khiển**
> (cách tối ưu tham số SMC có ràng buộc ổn định), **không** phải ở lane detection.

---

## 0. TL;DR — chạy nhanh nhất

```bash
cd journal_cmc
python run.py gpucheck                 # kiểm tra GPU + thư viện
# --- Mở Map_demo_v3.exe (nhấn đúp), để cửa sổ map chạy ---
python run.py --quick all              # chạy TOÀN BỘ pipeline (profile nhanh ~15-20 phút)
# xem kết quả trong results/tables/*.csv và results/figures/*.png
```
Khi đã ổn, chạy bản đầy đủ cho paper (nhiều giờ — xem Mục 6):
```bash
python run.py all
```

---

## 1. Kiến trúc & luồng dữ liệu

```
Map_demo_v3.exe (Unity, TCP server :54321)  ⇄  run.py (client)
    server → { Speed, Angle, Img(base64 jpg) }   mỗi frame
    client → "steer_angle  speed_cmd"

  perception.py   ảnh → lane error (px) [Hough]  (+ tiêm nhiễu noise/shadow/occlusion/blur)
  controller.py   error → lệnh lái/ga  [Boundary-Layer SMC hoặc PID]
  simulator.py    vòng lặp 1 episode (settle → measure), 1 kết nối cho NHIỀU trial (Option A)
  metrics.py      log 1 episode → mọi chỉ số (RMSE, SSI, jerk, chattering, Lyapunov…)
  objective.py    J(θ) = Σ wᵢ·metricᵢ/refᵢ  (ref lấy từ Manual-SMC)  ← optimizer tối thiểu hoá
  optimizers.py   BO(GPU)/PSO/Grid/Random  đề xuất θ
  experiments.py  4 thí nghiệm → lưu results/optim/*.json
  analysis.py     JSON → 7 bảng (CSV+LaTeX) + 5 hình  (chạy offline)
```

**Vì sao Option A (1 exe, trial tuần tự) hợp lệ:** tham số điều khiển chỉ là biến
Python. Giữ **một** kết nối, đổi controller/θ giữa các trial; xe vẫn chạy trên track
lặp vòng; mỗi trial có `T_SETTLE` giây "settle" để hấp thụ transient khi chuyển θ.
→ Không cần mở lại .exe hàng trăm lần.

---

## 2. Cài đặt

Máy bạn đã có sẵn: Python 3.10, numpy, opencv, scipy, pandas, matplotlib,
**torch 2.7.1+cu126 + RTX 3060**. Chỉ cần cài thêm 3 gói (đã cài trong phiên này):

```bash
pip install scikit-optimize pyswarms botorch
```
Kiểm tra:
```bash
python run.py gpucheck
```
Kỳ vọng (đã xác nhận trên máy bạn):
```
torch: 2.7.1+cu126 | CUDA available: True
GPU  : NVIDIA GeForce RTX 3060 Laptop GPU
botorch  : OK 0.16.1
skopt    : OK 0.10.2
pyswarms : OK 1.3.0
BO backend that will be used: botorch      ← BO chạy trên GPU
```

---

## 3. ⚠️ GPU & thời gian chạy — đọc kỹ (trung thực)

- **Mô phỏng chạy REAL-TIME.** Mỗi episode tốn đúng `T_SETTLE + T_MEASURE` giây
  đồng hồ thật, **bất kể GPU**. Tổng thời gian một thí nghiệm ≈ *(số episode) ×
  (thời lượng episode)*. GPU **không** rút ngắn được phần này — đây là bản chất
  của việc điều khiển một sim thời gian thực qua socket.
- **GPU dùng vào đâu:** surrogate Gaussian Process của **Bayesian Optimization**
  (BoTorch) fit trên RTX 3060 (`device=cuda`). Với 6 chiều/≤100 lần đánh giá,
  phần GP này rất nhẹ; GPU được dùng đúng nghĩa nhưng **không** phải nút thắt.
- Nếu muốn ép CPU cho BO: sẽ tự fallback sang `scikit-optimize` khi không có
  botorch/CUDA. Không cần chỉnh gì.
- **Kết luận thành thật để ghi vào Limitations:** thời gian chạy bị chi phối bởi
  sim real-time; GPU tăng tốc lớp tối ưu chứ không tăng tốc vòng mô phỏng.

---

## 4. Quy trình chạy TAY (từng bước)

### Bước 1 — Mở map
Nhấn đúp `Map_demo_v3/Window/Map_demo_v3.exe`. Để cửa sổ Unity chạy (xe đứng chờ
lệnh từ client). **Chỉ mở 1 lần** cho cả loạt lệnh bên dưới.

### Bước 2 — Mở terminal tại thư mục journal_cmc
```bash
cd d:/Xetuhanh/UTE_Car_2025/Map_demo____1/Window/cdoe/SMC_Hough_CMC/journal_cmc
```

### Bước 3 — Chạy thử 1 episode (kiểm tra kết nối + xem trực quan)
```bash
python run.py demo --method ManualSMC --show
```
- `--show`: hiện cửa sổ OpenCV vẽ lane + lệnh lái (nhấn `q` để dừng sớm).
- In ra bảng metric cuối episode. Nếu chạy được → kết nối OK.

> Nếu báo `Cannot connect to sim server` → chưa mở .exe hoặc sai cổng. Xem Mục 8.

### Bước 4 — Chạy pipeline
Có 2 cách:

**(a) Một phát ăn ngay** — chạy toàn bộ theo đúng thứ tự phụ thuộc:
```bash
python run.py all              # hoặc  python run.py --quick all  để thử nhanh
```
Thứ tự nội bộ: `reference → optimize(all) → compare → robustness → ablation → analyze`.

**(b) Từng bước** (khuyến nghị khi làm paper, để kiểm tra từng chặng):
```bash
python run.py reference        # [1] chuẩn hoá: chạy Manual-SMC, lưu reference_scales.json
python run.py optimize --which all   # [2] tối ưu: random, grid, pso, bo(GPU)  → Table 2,6 + Fig 2
python run.py compare          # [3] so sánh 6 method × nhiều seed       → Table 3 + Fig 3,5
python run.py robustness       # [4] nhiễu thị giác × method             → Table 4 + Fig 4
python run.py ablation         # [5] 5 biến thể ±boundary ±Lyapunov      → Table 5
python run.py analyze          # [6] dựng toàn bộ bảng + hình từ JSON (offline, không cần .exe)
```
Bạn có thể chạy `analyze` bất cứ lúc nào; nó dựng được bảng nào đã có dữ liệu và
bỏ qua (in "skipped") bảng chưa có.

> **Thêm `--quick`** vào bất kỳ lệnh sim nào để chạy profile nhanh (episode ngắn,
> ít seed) — dùng để smoke-test. Bỏ `--quick` = profile đầy đủ cho paper.

---

## 5. Từng lệnh + OUTPUT DỰ KIẾN

### `python run.py reference`
Chạy Manual-SMC 1 episode, lưu hệ số chuẩn hoá objective.
```
[reference] Manual-SMC baseline metrics recorded. Normalisation scales:
    lane_rmse_px   = ...
    ssi            = ...
    ...
```
→ ghi `results/optim/reference_scales.json`, `results/logs/reference_manualSMC.json`.

### `python run.py optimize --which all`
```
[optimize:random] done in ...s | N evals | best J=... | backend=numpy device=cpu
[optimize:grid]   done in ...s | ...
[optimize:pso]    done in ...s | ... | backend=pyswarms device=cpu
[optimize:bo]     done in ...s | ... | best J=... | backend=botorch device=cuda   ← GPU
    best gains: {'lambda_a': ..., 'eta_a': ..., ...}
[optimize:pid]    done in ...s | ... | best J=... | backend=botorch device=cuda   ← tuned PID
    best gains: {'Kp_angle': ..., 'Ki_angle': ..., ...}
```
→ mỗi optimizer ghi `results/optim/<name>_result.json` (best θ, history, running_best,
wall_time, backend/device, kind) và `<name>_history.json` (đầy đủ metric mỗi lần đánh giá).
`--which all` chạy cả 4 optimizer SMC **và** BO-tuned PID. Chạy riêng: `--which pid`.

### `python run.py compare`
```
[compare] seed 0 | order=['PID', 'TunedPID', 'ManualSMC', 'GridSMC', 'PSO_SMC', 'BO_SMC']
[compare] seed 1 | order=['TunedPID', 'ManualSMC', 'GridSMC', 'PSO_SMC', 'BO_SMC', 'PID']
...
[compare] saved results/optim/compare_results.json
```
→ 6 method (gồm **TunedPID**), thứ tự **interleave xoay vòng theo seed** (công bằng vị trí
track + pairing hợp lệ cho Table 7). Mỗi method: metric theo từng seed. Lưu 1 log đại
diện/method cho Fig 3/5.

### `python run.py robustness`
```
[robust] PID / clean: 10 seeds ...
[robust] PID / noise: ...
[robust] BO_SMC / occlusion: ...
[robust] saved results/optim/robustness_results.json
```
→ {PID, ManualSMC, BO_SMC} × {clean, noise, shadow, occlusion, blur}.

### `python run.py ablation`
```
[ablation] ManualSMC (switch=conference, lyap=False, optimize=False)
[ablation] BO_noBL_noLyap (switch=sign, lyap=False, optimize=True)
[ablation] BO_BL_noLyap   (switch=sat,  lyap=False, optimize=True)
[ablation] BO_noBL_Lyap   (switch=sign, lyap=True,  optimize=True)
[ablation] BO_full        (switch=sat,  lyap=True,  optimize=True)
```
→ mỗi biến thể được **tối ưu lại** với switch/objective tương ứng rồi đánh giá.

### `python run.py analyze`
```
=== TABLES ===
[table2] ... wrote results/tables/table2_parameters.csv
[table3] ... [table4] ... [table5] ... [table6] ... [table7] ...
=== FIGURES ===
    wrote results/figures/fig2_convergence.png
    wrote results/figures/fig3_steering.png
    wrote results/figures/fig4_lateral_occlusion.png
    wrote results/figures/fig5_tradeoff.png
```

**Bảng ↔ file:**
| Bảng paper | File |
|---|---|
| Table 1 (conference vs journal) | *viết tay trong paper* — mô tả |
| Table 2 (optimized params) | `table2_parameters.csv/.tex` |
| Table 3 (overall performance) | `table3_overall.csv/.tex` |
| Table 4 (robustness) | `table4_robustness.csv/.tex` |
| Table 5 (ablation) | `table5_ablation.csv/.tex` |
| Table 6 (optimizer efficiency) | `table6_optimizer_efficiency.csv/.tex` |
| Table 7 (significance) | `table7_significance.csv/.tex` |

**Hình ↔ file:** Fig 2 = `fig2_convergence.png`, Fig 3 = `fig3_steering.png`,
Fig 4 = `fig4_lateral_occlusion.png`, Fig 5 = `fig5_tradeoff.png`. (Fig 1 = sơ đồ
control loop, vẽ tay.)

---

## 6. Thời gian chạy dự kiến (profile FULL)

Với `T_SETTLE=6s`, `T_MEASURE=45s` → **~51s/episode** (chỉnh trong `config.py`).

| Bước | Số episode | Ước lượng |
|---|---:|---:|
| reference (× N episode) | 3 | ~3 phút |
| optimize (random 40 + grid 27 + pso 80 + bo 60 + **pid 60**) | ~267 | ~3.8 giờ |
| compare (**6** method × 15 seed) | 90 | ~1.3 giờ |
| robustness (3 × 5 × 10) | 150 | ~2.1 giờ |
| ablation (4 BO×60 + 4×10 eval + manual) | ~282 | ~4 giờ |
| **Tổng** | ~792 | **~11 giờ** |

**Cách rút ngắn (chỉnh `config.py`):**
- Giảm `T_MEASURE_FULL` xuống 30s, `BUDGETS['full']['bo_n_calls']` xuống 40.
- Chạy `ablation` với budget nhỏ hơn (nó tốn nhất vì tối ưu lại 4 lần).
- Giảm `compare_seeds`/`robustness_seeds` xuống 8–10 (đủ cho kiểm định).
- Có thể chạy rải nhiều buổi: mỗi lệnh lưu JSON riêng, `analyze` gom lại sau.

> Mẹo: chạy `--quick all` trước (~15–20 phút) để chắc mọi thứ thông, rồi mới chạy FULL.

---

## 7. Lưu trữ & tái sử dụng

Mọi thứ nằm trong `results/` (JSON là nguồn chân lý):
```
results/
  logs/     *.json   quỹ đạo thô từng episode (t, lane_error, angle, speed, s, ds…)
  optim/    *.json   reference_scales, <opt>_result/_history, compare/robustness/ablation_results
  tables/   *.csv .tex   7 bảng
  figures/  *.png    5 hình
```
- **Dựng lại bảng/hình bất cứ lúc nào** mà không cần chạy sim: `python run.py analyze`.
- **Không mất dữ liệu:** mỗi thí nghiệm ghi file riêng; chạy lại chỉ ghi đè file của
  chính nó. Muốn giữ nhiều lần chạy → copy thư mục `results/` ra chỗ khác trước.
- File `<opt>_history.json` giữ **đầy đủ metric mỗi lần optimizer đánh giá** → phân
  tích sâu thêm (vẽ landscape, so sánh acquisition…) mà không chạy lại.

---

## 8. Test KHÔNG cần mở .exe (mock server)

Muốn kiểm tra code/pipeline khi không tiện mở Unity:
```bash
# Terminal 1:
python mock_server.py
# Terminal 2:
python run.py --quick all
```
`mock_server.py` giả lập đúng giao thức TCP/JSON, render lane vàng tổng hợp + mô hình
xe bám-làn đơn giản. **Chỉ để kiểm thử pipeline — KHÔNG dùng số này cho paper.** Số
liệu paper phải lấy từ `Map_demo_v3.exe` thật.

---

## 9. Khắc phục sự cố

| Triệu chứng | Nguyên nhân / xử lý |
|---|---|
| `Cannot connect to sim server ...:54321` | Chưa mở `Map_demo_v3.exe`, hoặc cổng khác → `--port`. |
| `sim server closed the connection` | Unity bị đóng/crash giữa chừng → mở lại .exe, chạy lại lệnh. |
| BO in `backend=skopt device=cpu` | Thiếu botorch hoặc CUDA → `pip install botorch`, kiểm tra `gpucheck`. |
| `No optimizer result at ...` khi chạy compare | Chưa chạy `optimize` trước (compare cần best gains). |
| Bảng ra `n/a` nhiều | Episode quá ngắn / mất lane liên tục → tăng `T_MEASURE`, kiểm tra ngưỡng Hough. |
| `lane_dev_pct` = None/nan | Không ước lượng được lane width khung đó → chỉ là xấp xỉ, không ảnh hưởng RMSE. |
| Muốn đổi tốc độ mục tiêu / clip | Sửa `SPEED_TARGET`, `STEER_CLIP`, `SPEED_CLIP` trong `config.py`. |

---

## 10. 4 điểm PHẢI trung thực khi viết paper (tránh reviewer bắt lỗi)

1. **Đơn vị lệch làn = PIXEL** (và % lane-width xấp xỉ). Sim không trả vị trí thật →
   khai báo rõ trong Limitations, đừng ghi "Lateral RMSE (m)".
2. **Lyapunov = "inspired penalty"** (phạt `s·Δs > 0`), **không** phải chứng minh ổn
   định closed-loop (không có plant model). Giữ đúng chữ "inspired/regularizer".
3. **Baseline Manual-SMC là cấu hình conference** (η rất nhỏ ⇒ gần P-control). Cải
   thiện lớn là do **tối ưu hoá**; nêu trung thực, đừng để trông như dìm baseline.
4. **Chỉ có 1 track (`level0`).** "Robustness" đến từ **nhiễu thị giác tiêm trong
   Python**, không phải nhiều bản đồ khác nhau. Đừng claim "nhiều môi trường đường".

---

## 10b. Đã qua review đối kháng đa tác nhân (16 findings)

Bộ code đã được rà bằng review 5 chiều (runtime / control-theory / thống kê / thiết
kế / tối ưu số học), mỗi phát hiện được verify đối kháng. **Các lỗi validity đã được
VÁ trong code:**

| Đã sửa trong code | Ở đâu |
|---|---|
| Frame **mất lane không còn bị tính `error=0`** (trước đây làm đẹp giả RMSE); nay impute lỗi cỡ lane-departure, thêm `lane_rmse_detected` minh bạch | `metrics.py` |
| BO **không nạp điểm fail vào GP** (sentinel 1e3 cũ làm hỏng standardize); hạ `FAILURE_OBJECTIVE=50` | `optimizers.py`, `config.py` |
| **Interleave** thứ tự method theo từng seed → so sánh công bằng vị trí track + pairing hợp lệ cho Table 7 | `experiments.py` |
| Seed **thực sự đổi episode** (warm-up ngẫu nhiên theo seed) → hết pseudo-replication | `simulator.py` |
| Table 7 dùng **1 test a priori (Wilcoxon) + hiệu chỉnh Holm**, bỏ `min(p)` (chống p-hacking) | `analysis.py` |
| **Floor reference** nhất quán → 1 term không bị nổ khi baseline ~0 | `objective.py` |
| Table 6 thêm cột **Fails/Dims**; reference trung bình **N episode**; recv gộp chống phân mảnh TCP | nhiều file |

**Các caveat CÒN LẠI — xử lý bằng cách viết đúng trong paper (không phải bug code):**

5. **✅ ĐÃ GIẢI QUYẾT — có baseline "Tuned PID".** Trước đây PID chỉ ở cấu hình conference
   (chưa tune) nên reviewer có thể hỏi "proposed thắng PID do *tune* hay do *cấu trúc
   SMC*?". Nay đã tích hợp **`TunedPID`**: tune PID bằng **chính BO + chính objective** như
   SMC (`optimize --which pid`, tự chạy trong `optimize --which all`). Table 3 có cả *PID
   (untuned)* lẫn *Tuned PID (BO)*; **Table 7 có phép so sánh then chốt `BO_SMC vs TunedPID`**
   — cả hai đều đã tune, nên nếu SMC vẫn thắng thì đó là nhờ **cấu trúc**, không phải tuning.
   Kết hợp với Table 5 (ablation) → tách bạch hoàn toàn "cấu trúc" khỏi "tuning".
6. **SSI / steer_jerk tính theo FRAME, không theo giây.** So sánh giữa các method vẫn công
   bằng vì cùng fps; nhưng nên **báo cáo fps** và nêu rõ đơn vị. `chatter_tv` (tổng biến
   thiên/giây) là bản per-second đã có sẵn nếu reviewer yêu cầu.
7. **Tuning dùng `OPT_EVAL_SEEDS=1` episode/ứng viên** (mặc định, để tiết kiệm thời gian).
   Với bài cuối, đặt `OPT_EVAL_SEEDS=2–3` trong `config.py` để J bớt nhạy nhiễu real-time
   (đổi lại chậm hơn 2–3×). Ngoài ra `phi` phụ thuộc fps → lưu kèm phân bố fps khi báo cáo
   best-gains để tái lập được.

> Tất cả điểm trên đã ghi thành comment ngay trong code tại chỗ liên quan.

---

## 11. Bản đồ file

```
journal_cmc/
  config.py        tham số, search space, trọng số J, budget, đường dẫn
  perception.py    Hough lane detection + tiêm nhiễu + ước lượng lane width
  controller.py    Boundary-Layer SMC (sat/sign/conference) + PID conference
  metrics.py       log episode → mọi chỉ số (4 nhóm + Lyapunov)
  simulator.py     SimClient (socket) + run_episode (settle→measure)  [Option A]
  objective.py     J(θ), chuẩn hoá theo reference, Evaluator (ghi history)
  optimizers.py    bayes_opt(botorch GPU/skopt), pso, grid_search, random_search
  experiments.py   set_reference / run_optimize / run_compare / run_robustness / run_ablation
  analysis.py      dựng 7 bảng (CSV+LaTeX) + 5 hình từ JSON  [offline]
  run.py           CLI: gpucheck|demo|reference|optimize|compare|robustness|ablation|analyze|all
  mock_server.py   server giả lập để test không cần .exe
  requirements_journal.txt
  results/         logs / optim / tables / figures   (dữ liệu tái sử dụng)
```
