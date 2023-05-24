import os
import multiprocessing as mp
from functools import partial
from syntheticImageGenerator import SyntheticNutritionLabelImageGenerator
from tqdm.contrib.concurrent import process_map

def main():
    generator = SyntheticNutritionLabelImageGenerator()

    train = False
    num_image = 5000 if train else 500
    num_cores = mp.cpu_count()
    result_images_file_dir = os.path.join("result", "train" if train else "test")
    ground_truth_label_file_name = "train.txt" if train else "test.txt"
    # with mp.Pool(num_cores) as p:
    image_paths = [os.path.join(result_images_file_dir, f"{i}.png") for i in range(num_image)]
    process_map(partial(generator, ground_truth_label_file_name=ground_truth_label_file_name) , image_paths, max_workers=num_cores)
    # for i in range (num_image):
    #     ground_truth_label_file_name = "train.txt" if train else "test.txt"
    #     image_path = os.path.join(result_images_file_dir, f"{i}.png")
    #     generator(image_path, ground_truth_label_file_name)
    with open('result/' + ground_truth_label_file_name, "r") as f:
        lines = f.readlines()
    with open('result/' + ground_truth_label_file_name, "w") as f:
        for line in lines:
            if line.strip('\n').split(' ')[1].count(',') == 4 and \
            not line.strip('\n').endswith(','):
                f.write(line)

if __name__ == "__main__":
    # main()
    generator = SyntheticNutritionLabelImageGenerator()
    generator.illustrate_process()

