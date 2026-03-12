# 🚗 Vietnamese License Plate Recognition

Hệ thống nhận diện biển số xe Việt Nam sử dụng AI — phát hiện và đọc biển số từ ảnh tải lên.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-PP--OCRv4-0052CC?logo=paddlepaddle&logoColor=white)

---

## 📋 Tổng quan

Hệ thống sử dụng pipeline AI hoàn chỉnh để nhận diện biển số xe Việt Nam:

```
Upload ảnh → Phát hiện biển số (YOLOv8) → Cắt biển số → Nhận diện ký tự (PaddleOCR) → Hậu xử lý (Regex) → Hiển thị kết quả
```

### Đối tượng sử dụng
- Nhân viên quản lý bãi đỗ xe
- Nhân viên giám sát giao thông
- Lập trình viên kiểm thử mô hình AI

---

## 🏗️ Kiến trúc hệ thống

```
LicensePlateDetection/
│
├── app.py                          # Ứng dụng Streamlit chính (UI)
├── requirements.txt                # Danh sách thư viện
├── README.md
│
├── models/
│   └── plate_yolo.pt               # Model YOLOv8 đã train cho biển số xe
│
├── src/                            # Các module xử lý chính
│   ├── detect_plate.py             # Phát hiện biển số bằng YOLOv8
│   ├── crop_plate.py               # Cắt vùng biển số từ bounding box
│   ├── ocr_plate.py                # Đọc ký tự bằng PaddleOCR
│   └── plate_postprocess.py        # Hậu xử lý & format biển số VN
│
└── .streamlit/
    └── config.toml                 # Cấu hình Streamlit (nếu có)
```

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| **YOLOv8** (Ultralytics) | Phát hiện vị trí biển số trong ảnh |
| **PaddleOCR** | Nhận diện ký tự trên biển số |
| **OpenCV** | Xử lý ảnh, cắt biển số, vẽ bounding box |
| **Regex** | Hậu xử lý, format biển số theo chuẩn VN |
| **Streamlit** | Giao diện web tương tác |
| **Python 3.10+** | Ngôn ngữ lập trình chính |

---

## ⚙️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/hhtuann/LicensePlateDetection.git
cd LicensePlateDetection
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 🚀 Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

---

## 📖 Hướng dẫn sử dụng

### Bước 1 — Upload ảnh
Mở sidebar bên trái, chọn **Upload image / video** để tải ảnh chứa xe / biển số.

### Bước 2 — Nhận diện
Nhấn nút **Bắt đầu nhận diện**. Hệ thống sẽ chạy pipeline:
1. **Detect** — YOLOv8 phát hiện các biển số trong ảnh
2. **Crop** — Cắt từng biển số ra khỏi ảnh gốc
3. **OCR** — PaddleOCR đọc ký tự trên mỗi biển số
4. **Format** — Regex format thành biển số VN chuẩn (VD: `29Y3-78729`)

### Bước 3 — Xem kết quả
- Ảnh chính hiển thị bounding box quanh từng biển số
- Danh sách thumbnail biển số hiển thị bên dưới ảnh chính
- Nhấn **Xem chi tiết** để xem thông tin chi tiết từng biển số:
  - Ảnh crop biển số
  - Số biển đã format
  - Raw OCR output
  - Confidence score
  - Bounding box coordinates

---

## 🔧 Chi tiết các module

### `src/detect_plate.py` — PlateDetector
```python
class PlateDetector:
    def __init__(self, model_path)    # Load YOLOv8 model
    def detect(self, img) -> list     # Trả về list {"bbox", "conf"}
```
- Input: ảnh BGR (numpy array)
- Output: danh sách dict chứa `bbox` (x1, y1, x2, y2) và `conf` (float)

### `src/crop_plate.py` — crop_plate
```python
def crop_plate(img, bbox, resize_width=320) -> numpy.ndarray
```
- Cắt vùng biển số theo bounding box
- Resize về chiều rộng 320px (giữ tỉ lệ)

### `src/ocr_plate.py` — PlateOCR
```python
class PlateOCR:
    def __init__(self)                # Khởi tạo PaddleOCR (lang="en")
    def read_text(self, img) -> list  # Trả về list các chuỗi ký tự
```
- Input: ảnh biển số đã crop (numpy array)
- Output: danh sách text đọc được

### `src/plate_postprocess.py` — parse_plate
```python
def parse_plate(texts) -> str | None
```
- Nhận list text từ OCR
- Clean text (uppercase, loại bỏ ký tự đặc biệt)
- Tìm biển 1 dòng: `{2 số tỉnh}{1-2 chữ seri}{4-5 số}` → format `XXY-XXXXX`
- Tìm biển 2 dòng: ghép mã tỉnh + seri + cụm số
- Trả về biển số đã format hoặc `None`

---

## 🖥️ Giao diện

| Thành phần | Mô tả |
|---|---|
| **Sidebar trái** | Upload ảnh, nút nhận diện, trạng thái hệ thống |
| **Khu vực chính** | Hiển thị ảnh gốc với bounding box |
| **Stats row** | Số biển phát hiện, biển đang chọn, confidence trung bình |
| **Thumbnail strip** | Danh sách biển số thu nhỏ, nhấn để xem chi tiết |
| **Detail card** | Thông tin chi tiết biển số được chọn |

### Tính năng UI
- Giao diện hiện đại, gradient background
- Hỗ trợ ảnh chứa nhiều biển số
- Hiển thị confidence score cho từng biển số
- Highlight biển số đang chọn trên ảnh gốc
- Loading animation khi đang xử lý
- Responsive layout

---

## 📁 Dữ liệu mẫu

Đặt ảnh test vào thư mục bất kỳ hoặc upload trực tiếp qua giao diện. Ảnh nên:
- Chứa xe có biển số rõ ràng
- Định dạng: JPG, JPEG, PNG
- Biển số Việt Nam (xe máy, ô tô)

---

## ⚠️ Lưu ý

- Model `plate_yolo.pt` cần được train trước trên dataset biển số xe VN
- PaddleOCR sử dụng `lang="en"` vì biển số VN chỉ chứa ký tự Latin và số
- Regex hậu xử lý hiện hỗ trợ format biển số xe máy/ô tô phổ biến
- Chất lượng nhận diện phụ thuộc vào:
  - Độ phân giải ảnh
  - Góc chụp biển số
  - Điều kiện ánh sáng
  - Tình trạng biển số (mới/cũ, bẩn/sạch)

---

## 📄 License

Dự án được phát triển phục vụ mục đích học tập và nghiên cứu tại **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**.

