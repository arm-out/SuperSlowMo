import os
import random
import shutil
from pathlib import Path

VIDEO_EXT = {'.m4v', '.mov', '.MOV', '.mp4'}
INPUT_DIR = Path(
    "/home/armout/Documents/dev/superslowmo/original_high_fps_videos")
OUTPUT_DIR = Path("/home/armout/Documents/dev/superslowmo/frames")
WIDTH = 640
HEIGHT = 360

# Extract frames from videos


def extract_frames(in_dir, out_dir, width, height):
    for video in in_dir.glob("**/*"):
        if video.suffix in VIDEO_EXT:
            output_file = out_dir / video.name
            Path(output_file).mkdir(parents=True, exist_ok=True)

            cmd = "ffmpeg -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg".format(
                video, width, height, output_file
            )
            os.system(cmd)


# Group frames in batch sizes
def group_frames(in_dir, out_dir, n_frames=12):
    folders = 0
    for folder in in_dir.glob("**"):
        files = sorted(f for f in folder.glob("*") if f.is_file())
        acc = []
        for file in files:
            acc.append(file)
            if len(acc) >= n_frames:
                Path("{}/{}".format(out_dir, folders)).mkdir(
                    parents=True, exist_ok=True
                )
                for a in acc:
                    shutil.move(
                        str(a), "{}/{}/{}".format(out_dir,
                                                  folders, a.name)
                    )
                acc = []
                folders += 1


if __name__ == "__main__":
    train_dir = INPUT_DIR / "train"
    train_out = OUTPUT_DIR / "train"
    tmp = train_out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    extract_frames(train_dir, tmp, str(WIDTH), str(HEIGHT))
    group_frames(tmp, train_out)
    shutil.rmtree(tmp)

    test_dir = INPUT_DIR / "test"
    test_out = OUTPUT_DIR / "test"
    tmp = test_out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    extract_frames(test_dir, tmp, str(WIDTH), str(HEIGHT))
    group_frames(tmp, test_out)
    shutil.rmtree(tmp)

    test_files = [folder for folder in test_out.glob("**")]
    sampled = random.sample(range(len(test_files)), 100)
    val_dir = OUTPUT_DIR / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    for s in sampled:
        shutil.move("{}/{}".format(test_out, s),
                    "{}/{}".format(val_dir, s))
