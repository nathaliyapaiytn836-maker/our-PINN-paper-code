import h5py
import os

# 确保路径和你电脑上的一致
file_path = 'Data/circle.mat' 

if not os.path.exists(file_path):
    print(f"找不到文件！请检查路径：{file_path}")
else:
    print("成功读取高保真 v7.3 MAT文件！")
    with h5py.File(file_path, 'r') as f:
        print("包含的变量有：", list(f.keys()))
        print("-" * 30)
        for k in f.keys():
            # 打印每个变量的形状
            print(f"变量名: {k:10} | 形状: {f[k].shape}")