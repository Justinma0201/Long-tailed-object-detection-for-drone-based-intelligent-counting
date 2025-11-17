import os
import glob

input_folder = './TAICA_CVPDL_2025_HW2/CVPDL_hw2/CVPDL_hw2/train'

output_file = './ground_truth.txt' # <--- 請修改這裡

# --------------------------

def merge_ground_truth(input_dir, output_path):
    
    search_pattern = os.path.join(input_dir, '*.txt')
    txt_files = glob.glob(search_pattern)
    
    txt_files.sort()
    
    if not txt_files:
        print(f"錯誤：在 '{input_dir}' 資料夾中沒有找到任何 .txt 檔案。")
        return

    print(f"找到了 {len(txt_files)} 個 .txt 檔案，將開始合併...")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            
            for image_index, file_path in enumerate(txt_files, start=1):
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        
                        for line in infile:
                            line = line.strip()
                            
                            if line:
                                new_line = f"{image_index},{line}\n"
                                
                                outfile.write(new_line)
                                
                except Exception as e:
                    print(f"讀取檔案 {file_path} 時發生錯誤: {e}")

        print(f"\n成功！所有資料已合併並儲存至: {output_path}")
        print(f"總共處理了 {len(txt_files)} 個檔案 (Image Index 從 1 到 {len(txt_files)})。")

    except IOError as e:
        print(f"寫入檔案 {output_path} 時發生錯誤: {e}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

if __name__ == "__main__":
    if not os.path.isdir(input_folder):
        print(f"錯誤：輸入資料夾 '{input_folder}' 不存在。")
    else:
        merge_ground_truth(input_folder, output_file)