# PAD Box Exporter

CLI Python xuất danh sách monster **owned** từ **video quay màn hình** Monster Book (Puzzle & Dragons). OCR label `No:XXXXX`, phân loại icon theo độ bão hòa màu, gộp frame trùng lặp → `box.json`.

---

## Workflow (video — khuyến nghị)

1. Monster Book → **Order: Num.**
2. Bật quay màn hình (iOS/Android), scroll **từ đầu đến cuối** với tốc độ vừa phải (~80% overlap giữa các lần dừng).
3. Không crop video; giữ nguyên full màn hình.
4. Export video (`.mp4` / `.mov`) và chạy CLI.

Quay video nhanh hơn nhiều so với chụp 300–400 screenshot thủ công.

---

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# hoặc: pip install -r requirements.txt
```

**Yêu cầu:** Python 3.11+. Không cần GPU.

| OCR backend | Platform | Ghi chú |
|---|---|---|
| `ocrmac` | macOS | Apple Vision — nhanh, khuyến nghị trên Mac |
| `easyocr` | Mọi OS | Fallback; tải model ~100MB lần đầu |

---

## Sử dụng

```bash
# Video screen recording (input chính)
pad-box-export scroll.mp4 -o box.json

# Verbose + báo cáo
pad-box-export scroll.mp4 -o box.json --report report.txt -v

# Tune sampling / thresholds
pad-box-export scroll.mp4 -o box.json \
  --frame-interval 0.35 \
  --sat-threshold 45 \
  --debug-crops ./debug/

# Folder ảnh (fallback / debug)
pad-box-export ./screenshots/ -o box.json
```

```bash
python -m pad_box_export scroll.mp4 -o box.json
```

**`command not found: pad-box-export`?** Package đã cài nhưng pyenv chưa tạo shim. Chạy một trong các cách sau:

```bash
pyenv rehash                                    # sau pip install
python -m pad_box_export scroll.mp4 -o box.json # luôn hoạt động
source .venv/bin/activate && pad-box-export ... # nếu dùng venv
```

### Flags

| Flag | Mô tả |
|---|---|
| `-o`, `--output` | File JSON đầu ra (default: `box.json`) |
| `--report` | Ghi `report.txt` |
| `-v`, `--verbose` | Log chi tiết |
| `--frame-interval` | Giây giữa các frame lấy mẫu (default `0.4`) |
| `--max-frames` | Giới hạn số frame (debug) |
| `--min-id` / `--max-id` | Giới hạn range monster ID |
| `--sat-threshold` | Ngưỡng saturation cho owned |
| `--white-threshold` | Ngưỡng pixel trắng ở giữa icon (`?` unknown) |
| `--debug-crops` | Lưu crop icon để debug |
| `--ocr-backend` | `auto` \| `ocrmac` \| `easyocr` |

Tool tự bỏ frame mờ (motion blur) và frame gần giống nhau để giảm thời gian xử lý.

---

## Output

```json
{
  "version": 1,
  "exported_at": "2026-06-12T10:00:00Z",
  "source": "monster_book_video",
  "owned": [13602, 13605, 13611],
  "meta": {
    "frames_processed": 180,
    "owned_count": 4521,
    "labels_seen": 9200,
    "duplicates_dropped": 4100,
    "entries_footer": 7744,
    "low_confidence_ids": [13614],
    "frame_interval_sec": 0.4
  }
}
```

---

## Kiến trúc

```
pad_box_export/
├── cli.py           # argparse entry
├── video.py         # extract & dedupe frames from recording
├── ocr.py           # No:XXXXX label detection
├── classifier.py    # owned vs seen/unknown (HSV heuristic)
├── dedupe.py        # merge across frames
├── pipeline.py      # orchestration
└── models.py        # BoxExport schema
```

**Pipeline:** video → sample frames → OCR → crop icon → classify → dedupe → `box.json`

---

## Phát triển

```bash
pip install -e ".[dev]"
pytest
```

---

## License

See [LICENSE](LICENSE).
