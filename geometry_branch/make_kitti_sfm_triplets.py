import argparse
import os


def parse_seqs(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def make_triplets(kitti_root: str, seqs, out_triplet_path: str):
    triplets = []
    for seq_id in seqs:
        img_dir = os.path.join(kitti_root, "sequences", seq_id, "image_2")
        if not os.path.isdir(img_dir):
            print(f"skip missing sequence dir: {img_dir}")
            continue
        img_names = sorted([name for name in os.listdir(img_dir) if name.endswith(".png")])
        for idx in range(1, len(img_names) - 1):
            prev_path = os.path.join(img_dir, img_names[idx - 1])
            cur_path = os.path.join(img_dir, img_names[idx])
            next_path = os.path.join(img_dir, img_names[idx + 1])
            triplets.append((prev_path, cur_path, next_path))

    os.makedirs(os.path.dirname(out_triplet_path) or ".", exist_ok=True)
    with open(out_triplet_path, "w") as f:
        for prev_path, cur_path, next_path in triplets:
            f.write(f"{prev_path} {cur_path} {next_path}\n")

    print(f"wrote {len(triplets)} odometry triplets to {out_triplet_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate strict KITTI odometry triplets from image_2 sequences.")
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--seqs", default="00,01,02,03,04,05,06,07,08,09,10")
    parser.add_argument("--out-triplet-path", default="kitti_sfm_triplets.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_triplets(args.kitti_root, parse_seqs(args.seqs), args.out_triplet_path)
