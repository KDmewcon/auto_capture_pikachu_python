# Pikachu Classic Pro Helper (Python)

Tool ho tro game Pikachu co dien:
- Chon man hinh va chon vung ban game
- Vung ROI da khoanh duoc dung truc tiep lam ban game
- Auto scan va nhan dien icon theo tung o
- Tim tat ca cap noi duoc theo luat toi da 2 goc
- O noi duoc se sang len, o khong noi duoc se toi di de de nhin nuoc choi
- Auto click cap hop le dau tien
- O vua click se tam bo qua trong vai nhan de tranh click lap
- Auto co cooldown theo cap vua bam de khong spam click lap khi man hinh cap nhat cham

## 1) Yeu cau

- Python 3.10+
- macOS da cap quyen Screen Recording cho Terminal/VS Code
- macOS da cap quyen Accessibility neu dung auto click
- Neu muon Esc global (khong can focus cua so preview), can cap quyen Accessibility cho Terminal/VS Code

## 2) Cai dat

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Chay tool

```bash
source .venv/bin/activate
python app.py
```

Khi chay lan dau, tool se:
1. Liet ke monitor
2. Cho chon monitor
3. Mo cua so de keo chuot chon vung ban Pikachu

## 4) Phim tat trong cua so preview

- `A`: Bat/tat auto mode
- `C`: Click 1 cap hop le
- `R`: Chon lai vung ban game
- `W`: Luu anh overlay hien tai
- `Esc`: Dung auto/click ngay lap tuc (khong thoat app)
- `Q`: Thoat

## 5) Tuy chon command line

```bash
python app.py --rows 9 --cols 16 --monitor 1 --interval 0.25 --auto-start
```

Them tham so tuning:
- `--empty-edge`
- `--empty-variance`
- `--empty-saturation`
- `--empty-ink`
- `--similarity`
- `--ambiguity-margin`
- `--template-update`
- `--max-lines`
- `--select-region`

Preset uu tien do chinh xac:

```bash
python app.py --monitor 1 --select-region \
	--similarity 0.90 --ambiguity-margin 0.01 --template-update 0.16
```

## 6) Test solver

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py" -v
```

## 7) Luu y

- File `pikachu_config.json` duoc tao tu dong de luu monitor/region/setting gan nhat.
- `board_grid` se tu dong bang toan bo vung ROI da khoanh.
- Tool gia dinh ban game la luoi deu (moi o cung kich thuoc).
- Neu nhan sai icon, hay tang `--similarity` va `--ambiguity-margin`.
- Neu hinh icon thay doi nhieu theo hieu ung man choi, co the tang `--template-update` nhe (0.18-0.25).
- Neu nhan sai o rong, tinh chinh bo 3 tham so `--empty-*`.
