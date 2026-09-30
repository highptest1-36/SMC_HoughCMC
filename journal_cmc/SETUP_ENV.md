# SETUP_ENV — Tạo môi trường & cài thư viện (1 lần, dùng mãi)

Mục tiêu: tạo **một môi trường conda riêng** tên `smc_cmc` cài sẵn đầy đủ (torch CUDA
cho RTX 3060 + tất cả thư viện). Sau này chỉ cần `conda activate smc_cmc` là chạy được,
không lo nhầm môi trường CPU như trước.

> Máy bạn đã có conda (dấu `(base)` ở terminal) + NVIDIA driver 610.47 → đủ điều kiện.

---

## ⚡ Cách nhanh nhất — chạy script tự động

Mở **Anaconda PowerShell Prompt**, `cd` vào thư mục này rồi:

```powershell
cd d:/Xetuhanh/UTE_Car_2025/Map_demo____1/Window/cdoe/SMC_Hough_CMC/journal_cmc
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

Script sẽ: tạo env `smc_cmc` (Python 3.10) → cài torch CUDA → cài mọi thư viện →
tự chạy `gpucheck`. Xong là dùng được. (Nếu muốn tên khác: `... -File .\setup_env.ps1 smc_cmc2`)

---

## 🔧 Cách thủ công — làm từng bước (nếu không muốn chạy script)

### Bước 1 — Tạo môi trường
```powershell
conda create -n smc_cmc python=3.10 -y
```

### Bước 2 — Kích hoạt
```powershell
conda activate smc_cmc
```
Prompt đổi thành `(smc_cmc)`. **Mọi bước sau làm trong env này.**

### Bước 3 — Cài torch CUDA (cho GPU 3060) — QUAN TRỌNG: cài torch TRƯỚC
```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```
> Phải cài từ `--index-url .../cu126` để lấy **bản CUDA**. Nếu cài `pip install torch`
> thường → ra bản CPU (đúng lỗi bạn gặp lúc nãy).

### Bước 4 — Cài các thư viện còn lại
```powershell
python -m pip install -r requirements_journal.txt
```

### Bước 5 — Kiểm tra
```powershell
python run.py gpucheck
```
Phải thấy:
```
torch: 2.7.1+cu126 | CUDA available: True
GPU  : NVIDIA GeForce RTX 3060 Laptop GPU
botorch  : OK 0.16.1
skopt    : OK 0.10.2
pyswarms : OK 1.3.0
BO backend that will be used: botorch      ← BO chạy GPU
```

---

## ✅ Mỗi lần chạy về sau (chỉ 2 dòng)

```powershell
conda activate smc_cmc
cd d:/Xetuhanh/UTE_Car_2025/Map_demo____1/Window/cdoe/SMC_Hough_CMC/journal_cmc
python run.py gpucheck        # (tuỳ chọn) xác nhận nhanh
# rồi mở Map_demo_v3.exe và chạy các lệnh trong QUICKSTART.md
```

Không cần cài lại gì. Muốn thoát env: `conda deactivate`.

---

## 🩹 Nếu KHÔNG cài được torch CUDA (mạng lỗi / muốn chạy tạm) — dùng CPU

Framework vẫn chạy tốt trên CPU (tự fallback sang `skopt`), và vì mô phỏng chạy
real-time nên **tốc độ gần như không đổi**. Cài bản CPU:
```powershell
conda activate smc_cmc
python -m pip install torch==2.7.1
python -m pip install -r requirements_journal.txt
python run.py gpucheck        # sẽ báo CUDA: False, backend: skopt — vẫn chạy được
```

---

## 🐍 Không dùng conda? Dùng venv thuần cũng được

```powershell
cd d:/Xetuhanh/UTE_Car_2025/Map_demo____1/Window/cdoe/SMC_Hough_CMC/journal_cmc
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements_journal.txt
python run.py gpucheck
```
Mỗi lần chạy: `.\.venv\Scripts\Activate.ps1`.

---

## 🔎 Xử lý sự cố

| Triệu chứng | Xử lý |
|---|---|
| `CUDA available: False` | Đã cài nhầm torch CPU → gỡ & cài lại bản index cu126: `pip uninstall torch -y` rồi Bước 3. |
| `conda: command not found` | Mở đúng **Anaconda PowerShell Prompt** (không phải PowerShell thường), hoặc `conda init powershell` rồi mở lại. |
| pip muốn **hạ/nâng torch** khi cài botorch | Đã cài torch trước + `botorch==0.16.1` đã pin nên không xảy ra; nếu vẫn bị, cài lại torch (Bước 3) sau cùng. |
| Tải torch quá lâu (~2.5GB) | Bình thường (bản CUDA nặng). Có thể dùng tạm bản CPU (mục 🩹) rồi cài CUDA sau. |
| `run.py` báo thiếu module | Chưa `conda activate smc_cmc`, hoặc đang ở env khác. |
| Muốn xoá làm lại | `conda remove -n smc_cmc --all -y` rồi tạo lại. |

---

## 📌 Ghi nhớ
- Env đúng để chạy GPU = **`smc_cmc`** (hoặc Python310 global — cũng đã cài sẵn cu126).
- Env **`base`** của bạn đang là **torch CPU** → đừng chạy trong `base` nếu muốn GPU.
- Kết quả lưu ra file JSON, **không phụ thuộc môi trường** — đổi env chỉ đổi GPU/CPU.
