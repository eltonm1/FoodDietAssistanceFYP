import cv2
import os
import synthetic
from tqdm.contrib.concurrent import process_map
import numpy as np
import shutil
from os.path import join
from multiprocessing import Process, Lock, Array, Value, Lock

class DataGenerator():
    def __init__(self, num_image, num_cores, train=True):
        self.num_image = num_image
        self.num_cores = num_cores
        self.train = train
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = join(self.current_path, "data", "train" if self.train else "val")
        self.cached_bboxes = {}

        # remove and re-create the folder directly
        shutil.rmtree(self.dir_path, ignore_errors=True)
        os.makedirs(self.dir_path)

        self.annotation_path = join(self.dir_path, "annotation.txt")
        # if exists, remove the file
        if os.path.exists(self.annotation_path): os.remove(self.annotation_path)

    def __call__(self):
        # create an array of string indices 
        indices = [i for i in range(self.num_image)]
        
        # generate images in parallel
        # process_map(
        #     self.save_image, 
        #     indices, 
        #     max_workers=self.num_cores
        #     )

        lock = Lock()
        for core in range(self.num_cores):
            Process(target=self.save_image, args=(indices[core::self.num_cores], lock)).start()

    def save_image(self, idxes, lock):
        for idx in idxes:
            final, bboxes = synthetic.generate()
            cv2.imwrite(join(self.dir_path, f"{idx}.png"), final)
            self.write_bboxes_to_txt(idx=idx, bboxes=bboxes, lock=lock)

    def write_bboxes_to_txt(self, idx, bboxes, lock):
        lock.acquire()
        with open(self.annotation_path, "a") as f:
            bboxes_str = " ".join([",".join([str(x) for x in bbox]) for bbox in bboxes])
            f.write(join(self.dir_path, f"{idx}.png") + " " + bboxes_str + "\n")
        lock.release()

if __name__ == "__main__":
    train = True
    num_image = 5000 if train else 100
    num_cores = int(os.cpu_count())
    generator = DataGenerator(num_image, num_cores, train)
    generator()
