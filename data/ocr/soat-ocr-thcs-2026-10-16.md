# Soát lỗi đọc chữ từ ảnh sách (OCR)

Số trang đã soát: 5941 (lấy mẫu ngẫu nhiên, seed = 2026 — chạy lại cùng seed sẽ ra đúng các trang này)
Số trang có công thức/phân số: 1990
Số trang bị gắn cờ nghi đọc sai: 99
Trong đó đã đọc lại bằng Gemini Vision và vá vào kho: 50
Còn phải xử lí: 49

Lưu ý: các cờ ở đây mô tả bản chữ do máy đọc tại chỗ (Tesseract), và bản đó không bị ghi đè — nên một trang đã vá vẫn còn cờ. Điều đã thay đổi là trong kho tìm kiếm đã có thêm một đoạn sạch cho trang đó.

## Thống kê theo loại lỗi

| Loại | Số lần |
|---|---|
| mat-so-mu | 50 |
| so-lan-vao-chu | 48 |
| dau-nhan-thanh-dau-tru | 2 |
| mat-hop-dinh-nghia | 2 |
| phan-so-mat-gach-ngang | 1 |

## Các trang cần mở sách đối chiếu

### Lớp 6 — SGKLichsuvaDiali6 — trang 4

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Bài2, Bài6, Bài9, Bài3`

### Lớp 6 — SGKTiengAnh6tap2GlobalSuccess — trang 71

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `ed3ukert, maœraô9an, sk2`

### Lớp 7 — SGKKhoahoctunhien7 — trang 26

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Ø2pdn, sượD6N, UạÁn6u, UBIð3, j3plMBE, œ2S, œ3`

### Lớp 7 — SGKLichsuvaDiali7 — trang 4

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `pai1g, Bài2, Bài3, Bài4, BÀH6, Bà7, Ball7`

### Lớp 7 — SGKLichsuvaDiali7 — trang 5

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Bài2, Bài3, Bài7`

### Lớp 7 — SGKNguvan7tapmot — trang 80

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Ý1, Ý2, Ý3`

### Lớp 7 — SGKTiengAnh7GlobalSuccess — trang 140

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `run0n, ss0la, telip2`

### Lớp 7 — SGKToan7taphai — trang 33

- **mat-so-mu**: Có 3 chỗ dạng 'A?' hoặc 'A*' — thường là số mũ (A², B³) bị máy đọc mất.
  - Máy đọc ra: `im tổng của hai đa thức: x° - 5x + 2 và x9 - x? + 6x - 4. Chú ý Phép cộng đa thức cũng có các tính chất như phép cộng các số thực. Cụ thể là: - Tính chất giao hoán: A + B = B + A. ~ Tính chất kết hợp: (A+ B) + Œ= A + (B + C). - Cộng với đa thức không: A +0 =0+A= A. Cho hai đa thức M = 0,5x* - 4x3 + 2x - 2,5 va N= 2x3 + x2 + 1,5. Hãy tính tổng M + N (trình bày theo hai cách). Đặt tính cộng để tìm tổng của ba đa thức sau: A= 2x8 - 5x*4.x-7; B=x2-2x+6; C=-x3+ 4x? - 1. B TRỪ HAI ĐA THỨC MỘT BIẾN y : 8 Hiệu của hai da thức Cho hai đa thức P = x* + 3x3 - 5x2 + 7x và Q= -x + 4x? - 2x + 1. Đối với phép trừ: P- Q = (xf + 3xŠ - 5x? + 7x) - (x3 + 4x2 - 2x + 1), ta cũng có hai cách trình bày, tương tự như phép cộng hai đa thức. Cụ thể, hãy thực hiện các hoạt động sau: |HĐ1] Tìm hiệu P - Q bằng cách bỏ dấu ngoặc rồi nhóm các hạng tử cùng bậc và thu gọn. [Z2 Tìm hiệu P - Q bằng cách đặt tính trừ: đặt đa thức Q dưới đa thức P sao cho các hạng tử cùng bậc thẳng cột với nhau rồi trừ theo từng cột. Cho hai đa thức: M = 0,5x4 - 4x3 + 2x - 2,5 và N= 2x9 + x? + 1,5. Hãy tính hiệu M - N (trình bày theo h`

### Lớp 7 — SGKToan7taphai — trang 49

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HĐ2, HĐÐ1`

### Lớp 7 — SGKToan7taphai — trang 79

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HD1, Hiihg3e, HĐ2`

### Lớp 7 — SGKToan7tapmot — trang 7

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HĐ2, vidu1`

### Lớp 8 — SGKAmnhac8 — trang 67

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Só2, Só8, s6i`

### Lớp 8 — SGKKhoahoctunhien8 — trang 104

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `vitri2, Pin3V, Pin6V`

### Lớp 8 — SGKLichsuvaDiali8 — trang 4

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Bài1, Bài2, Bài4, AnB6va, Bai8, Bài9, Bài1ạ`

### Lớp 8 — SGKLichsuvaDiali8 — trang 5

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `hương3, Bài9, Bài2, Bài3, Bài4, Bài5, Bài6, Bài7`

### Lớp 8 — SGKTiengAnh8GlobalSuccess — trang 109

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `cön4, AfY8M, Anh8`

### Lớp 8 — SGKTinhoc8 — trang 79

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `sb3`

### Lớp 8 — SGKToan8taphai — trang 20

- **mat-so-mu**: Có 3 chỗ dạng 'A?' hoặc 'A*' — thường là số mũ (A², B³) bị máy đọc mất.
  - Máy đọc ra: `6.21.a - ; b - : c) ——— 1 ere x+1 hx x+y re K?+1 1 2 3 12 3 6.22. a) — —— : ye xui X Da XI eke 2x-1, 1-x 3 1-2x, x-† 3 b) i # A x 2x+1 x2-9 x 2x+1 x+3 823.) XA, x 4X x?-4 2-x 5x-10 x 3 x-2ì 3 ( †1 x-2 b - ` > 4 lA (34) |x+8 lợn x2) 624.0) TƯ“, xy yz 2x x y ) + (x-yŸ` y?-x 6.25. Một tau du lịch chạy xuôi dòng 15 k`

### Lớp 8 — SGKToan8taphai — trang 41

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Thăng3, Tháng4, Tháng5`

### Lớp 8 — SGKToan8taphai — trang 42

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HĐ2, Vidu2`

### Lớp 8 — SGKToan8taphai — trang 49

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `À2, Vidu2, o4e`

### Lớp 8 — SGKToan8taphai — trang 102

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `AB2, đC2, AC2`

### Lớp 8 — SGKToan8taphai — trang 127

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HĐ2, ZZ7`

### Lớp 8 — SGKToan8tapmot — trang 5

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Bài4, Bai5, Bài3`

### Lớp 8 — SGKToan8tapmot — trang 7

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, XV4X, vídu1`

### Lớp 8 — SGKToan8tapmot — trang 9

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ3, HĐ4, HĐ6, HĐ5`

### Lớp 8 — SGKToan8tapmot — trang 10

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `xy2, x3y, x2y`

### Lớp 8 — SGKToan8tapmot — trang 17

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `x2y, xy2`

### Lớp 8 — SGKToan8tapmot — trang 18

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `x3y, x9y, x8y`

### Lớp 8 — SGKToan8tapmot — trang 38

- **mat-so-mu**: Có 5 chỗ dạng 'A?' hoặc 'A*' — thường là số mũ (A², B³) bị máy đọc mất.
  - Máy đọc ra: `ai số a, b bất kì, thực hiện phép tính (a+b)(z? ~ab+ bŸ, Từ đó rút ra liên hệ giữa a° + bŸ và (a + b)(a” ~ ab + b). Với A, B là hai biểu thức tuỳ ý, ta có: EXETED vist các a thúc sau duới dang tich: i a) x°+8; b) 8x3 + y3. - Giải a) xỶ+8= x+23 =(x+2)(x? ~2x +4). - b) 8xŸ+yŸ= (zx} +ự =(2x+y)(4x? ~exy+y*). | vidu2 | Rút gon các biểu thức sau: ị a) (x+3)(x? -3x +9)- x9; -b) (3x + v)(sx? -3xy +#) ~ y?- 26x. 3Ì`

### Lớp 8 — SGKToan8tapmot — trang 117

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `blelol4N, HĐ1`

### Lớp 9 — SGKCongnghe9TrainghiemnghenghiepModunTrongcayanqua — trang 32

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `PHS1, PHS2, HTM1, HTM2`

### Lớp 9 — SGKLichsuvaDiali9 — trang 5

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Bài1, Bài3, Bài4, Bài5, Bài6, Bài7, Bài8, Bài9, thửđể3`

### Lớp 9 — SGKNguvan9tapmot — trang 11

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `RMIH2, SS2, kiệt1`

### Lớp 9 — SGKTiengAnh9GlobalSuccess — trang 107

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `LOOK1, Anh9, mp0`

### Lớp 9 — SGKToan9taphai — trang 11

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HĐ2`

### Lớp 9 — SGKToan9taphai — trang 13

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `x2a, vidu4, xX2`

### Lớp 9 — SGKToan9taphai — trang 15

- **mat-so-mu**: Có 3 chỗ dạng 'A?' hoặc 'A*' — thường là số mũ (A², B³) bị máy đọc mất.
  - Máy đọc ra: `ðT „ _-7-v6† Giải các phương trình sau: a) x?-6x+9=0, b) 2x2+3x+5=0. Giai a) Ta có: A =(-) ~4-1-9=0. Do đó, phương trình có nghiệm kép: `... z5 ae ‘ b) Ta có: A= 3*-4.2.5=9-40=-31<0. Do đó, phương trình vô nghiệm. Áp dụng công thức nghiệm, giải các phương trình sau: a) 2x2~5x+1=0; b) x? +8x + 16 =0; e) x?-x+1=0. / 2 Thử thách nhỏ Có thể nói gì về`

### Lớp 9 — SGKToan9taphai — trang 22

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ1, HÐ2, HĐ2`

### Lớp 9 — SGKToan9taphai — trang 29

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `vidu2, cm3`

### Lớp 9 — SGKToan9taphai — trang 88

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ2, HĐ3, Hìnho4g`

### Lớp 9 — SGKToan9taphai — trang 109

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `cm2, cm3`

### Lớp 9 — SGKToan9taphai — trang 131

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `az0, ax2`

### Lớp 9 — SGKToan9tapmot — trang 22

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `cm3, HĐ1, HD2, HD1, HĐ2`

### Lớp 9 — SGKToan9tapmot — trang 47

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Vidu3, V2Ÿ, HĐ3, V2x`

### Lớp 9 — SGKToan9tapmot — trang 50

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `xÍ5, vidu2, xj5abŠ`

### Lớp 9 — SGKToan9tapmot — trang 57

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `xJB2, J4a, fZZ7`

### Lớp 9 — SGKToan9tapmot — trang 70

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `HĐ2, HĐ3, cosa0`

### Lớp 9 — SGKToan9tapmot — trang 106

- **so-lan-vao-chu**: Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu trang này quét kém.
  - Máy đọc ra: `Ø4, HĐ2`

## Cách xử lý

Mỗi dòng ở trên chỉ là NGHI VẤN. Cách làm: mở đúng trang sách giấy, đọc câu đó, nếu máy đọc sai thật thì thêm một mục vào `TEXT_FIXES` trong `src/dora_edu/data_pipeline/manual_patches.py` (sửa đúng chỗ sai), hoặc vào `PATCHES` (thêm một đoạn sạch), rồi chạy lại `dora-bulk-ingest` và khởi động lại bot.