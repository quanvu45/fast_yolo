import os
from pathlib import Path

input_file = '/home/ccne/Documents/YOLOMG/datasets/val2.txt'   # Đường dẫn tới file gốc
output_dir = '/home/ccne/Documents/YOLOMG/datasets/mask'  # Thư mục chứa các file output

os.makedirs(output_dir, exist_ok=True)

current_video = None
current_file = None

with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Trích tên video từ đường dẫn (phần thư mục cha của file ảnh)
        # Ví dụ: /home/.../images/phantom02/phantom02_0001.jpg -> phantom02
        video_name = Path(line).parent.name

        if video_name != current_video:
            # Đóng file cũ nếu đang mở
            if current_file:
                current_file.close()

            current_video = video_name
            out_path = os.path.join(output_dir, f"{video_name}.txt")
            current_file = open(out_path, "w")
            print(f"Tạo file: {out_path}")

        current_file.write(line + "\n")

if current_file:
    current_file.close()

print("Hoàn thành!")