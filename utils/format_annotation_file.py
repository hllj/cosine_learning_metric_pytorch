import os

DATA_DIR = "data/VRIC"


def format_csv(filename):
    f = open(os.path.join(DATA_DIR, filename), "r")
    name = filename.split(".")[0]
    f_csv = open(os.path.join(DATA_DIR, f"{name}.csv"), "w")
    f_csv.write("Image_name,ID,Cam\n")
    lines = f.readlines()
    f.close()
    for line in lines:
        img_filename, idx, cam_idx = line[:-1].split(" ")
        f_csv.write(f"{img_filename},{idx},{cam_idx}\n")
    f_csv.close()


if __name__ == "__main__":

    train_txt = "vric_train.txt"
    val_txt = "vric_gallery.txt"
    test_txt = "vric_probe.txt"

    format_csv(train_txt)
    format_csv(val_txt)
    format_csv(test_txt)
