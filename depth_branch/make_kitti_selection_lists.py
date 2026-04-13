import os
from glob import glob

def make_lists(
    root="",
    out_img_list="kitti_train_images.txt",
    out_depth_list="kitti_train_depths.txt",
):
    """
    使用 KITTI depth_selection 中的 val_selection_cropped 生成
    RGB 和深度的列表文件。

    目录结构假定为：
      root/
        image/
          0000000000.png
          ...
        groundtruth_depth/
          0000000000.png
          ...

    生成：
      out_img_list:  每行一个 RGB 绝对路径
      out_depth_list:每行一个深度绝对路径
    """
    img_dir = os.path.join(root, "image")
    depth_dir = os.path.join(root, "groundtruth_depth")

    img_files = sorted(glob(os.path.join(img_dir, "*.png")))
    depth_files = sorted(glob(os.path.join(depth_dir, "*.png")))

    assert len(img_files) > 0, f"在 {img_dir} 下没有找到 PNG 图像"
    assert len(img_files) == len(depth_files), \
        f"RGB 数量 {len(img_files)} != depth 数量 {len(depth_files)}"

    with open(out_img_list, "w") as fi, open(out_depth_list, "w") as fd:
        for img, dep in zip(img_files, depth_files):
            fi.write(os.path.abspath(img) + "\n")
            fd.write(os.path.abspath(dep) + "\n")

    print(f"写入 {out_img_list}, 共 {len(img_files)} 行")
    print(f"写入 {out_depth_list}, 共 {len(depth_files)} 行")

if __name__ == "__main__":
    make_lists()
