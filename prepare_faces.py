import os

import cv2
import fire
import numpy as np
import yaml
from tqdm import tqdm

from aligner import Alinger
from RetinaFace.retinaface import RetinaFace


def make_dir_safe(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rework_landmarks(landamrks):

    landamrks = landamrks.astype(int)
    new_lands = list(landamrks[:, 0])
    new_lands.extend(list(landamrks[:, 1]))
    return np.array([new_lands])


def main(config: str = "config.yml"):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    enter_folder = cfg["ent_folder"]
    out_folder = cfg["out_folder"]
    mode = cfg["mode"]
    make_dir_safe(out_folder)

    detector = RetinaFace(cfg["name_det"], cfg["epoch_det"], cfg["gpu"])
    align = Alinger(cfg["size"])

    for race in tqdm(os.listdir(enter_folder)):
        ent_race = os.path.join(enter_folder, race)
        out_race = os.path.join(out_folder, race)
        make_dir_safe(out_race)
        for gender in os.listdir(ent_race):
            ent_gender = os.path.join(ent_race, gender)
            out_gender = os.path.join(out_race, gender)
            make_dir_safe(out_gender)
            faces = os.listdir(ent_gender)
            faces.sort()
            k = 0
            out_faces = []
            imgs = []
            for face in tqdm(faces):
                k += 1
                ent_face = os.path.join(ent_gender, face)
                out_face = os.path.join(out_gender, face)
                img = cv2.imread(ent_face)
                try:
                    img = cv2.resize(img, (500, 500))
                    det_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, dets = detector.detect(det_img)
                except cv2.error:
                    dets = np.empty((0, 1))
                    print(ent_face)
                if dets.shape[0] != 1:
                    if mode == "unknown":
                        continue
                    imgs.append(0)
                else:
                    lands = rework_landmarks(dets[0])
                    rew_imgs = align([img], [lands])
                    if mode == "unknown":
                        cv2.imwrite(out_face, rew_imgs[0][0][0])
                        continue
                    imgs.append(rew_imgs[0][0][0])
                out_faces.append(out_face)
                if k == 2 and mode == "known":
                    if not isinstance(imgs[0], int) and not isinstance(imgs[1], int):
                        for img, path in zip(imgs, out_faces):
                            cv2.imwrite(path, img)
                    imgs = []
                    out_faces = []
                    k = 0


if __name__ == "__main__":
    fire.Fire(main)
