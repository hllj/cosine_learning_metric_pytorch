import os

DATA_DIR = "data/VRIC"


def format_csv(class_dict, class_id, filename):
    f = open(os.path.join(DATA_DIR, filename), "r")
    name = filename.split(".")[0]
    f_csv = open(os.path.join(DATA_DIR, f"{name}.csv"), "w")
    f_csv.write("Image_name,ID,Cam\n")
    lines = f.readlines()
    f.close()
    for line in lines:
        img_filename, idx, cam_idx = line[:-1].split(" ")
        if idx not in class_dict.keys():
          class_id += 1
          class_dict[idx] = class_id
        f_csv.write(f"{img_filename},{class_dict[idx]},{cam_idx}\n")
    f_csv.close()
    return class_dict, class_id


if __name__ == "__main__":
    class_dict = {}
    class_id = -1
    train_txt = "vric_train.txt"
    val_txt = "vric_gallery.txt"
    test_txt = "vric_probe.txt"

    class_dict, class_id = format_csv(class_dict, class_id, train_txt)
    class_dict, class_id = format_csv(class_dict, class_id, val_txt)
    class_dict, class_id = format_csv(class_dict, class_id, test_txt)
    print(class_id)
