import os
import argparse

def sum_int8_files_size(folder_path,key):
    total_size = 0
    
    for filename in os.listdir(folder_path):
        if key in filename.lower():
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    return total_size

def format_size(size_in_bytes):
    # 转换为 MB
    size_in_mb = size_in_bytes / (1024 * 1024)
    
    if size_in_mb >= 1024:
        # 如果大于等于1024MB，转换为GB并保留两位小数
        return f"{size_in_mb / 1024:.2f} GB"
    else:
        # 否则保留为MB并保留两位小数
        return f"{size_in_mb:.2f} MB"



def main():
    parser = argparse.ArgumentParser(description="memory")
    
    # 添加参数
    parser.add_argument('--error_path', type=str, help="error_path")
    parser.add_argument('--LoRC_dtype', type=str, help="LoRC_dtype")
    # 解析参数
    args = parser.parse_args()
    print(f"path: {args.error_path}")
    # 使用示例
    total_size = sum_int8_files_size(args.error_path,args.LoRC_dtype)
    formatted_size = format_size(total_size)
    print(f"包含{args.LoRC_dtype}的文件总大小为: {formatted_size}")

if __name__ == "__main__":
    main()