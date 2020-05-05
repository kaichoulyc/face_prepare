import math
from typing import List, Tuple

import cv2
import numpy as np


class Alinger:
    def __init__(self, face_size_side: int, padding: float = 0.3):
        """
        Aligner init.

        :param face_size_side: type(int) face size side.
        :param padding: type(float) scale of image padding.
        """
        self.face_size_side = face_size_side
        self.padding = padding

    @staticmethod
    def list2colmatrix(pts_list: list) -> np.matrix:
        """
        Convert list to column matrix

        Parameters:
        ----------
        :param pts_list: type(list) points.
        :return: colMat - col matrix
        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    @staticmethod
    def find_tfrom_between_shapes(
        from_shape: np.matrix, to_shape: np.matrix
    ) -> Tuple[np.matrix]:
        """
        Find transform between shapes

        Parameters:
        ----------
        :param from_shape: type(numpy.matrix
            Input shape
        :param to_shape: type(numpy.matrix
            Output shape
        :return: tran_m - transformation matrix, tran_b - transformation matrix
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0] // 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] // 2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (
                from_shape_points[i] - mean_from
            )

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(
        self, img: np.ndarray, points: np.ndarray
    ) -> List[np.ndarray]:
        """
        Crop and align face

        :param img: type(numpy.ndarray) image.
        :param points: type(numpy.ndarray) face points n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        Where x1, y1 = left eye; x2, y2 = right eye; x3, y3 = nose; x4, y4 = mouth left; x5, y5 = mouth right
        :return: crop_imgs - list of cropped and aligned faces
        """
        crop_imgs = []
        for p in points:
            shape = []
            for k in range(len(p) // 2):
                shape.append(p[k])
                shape.append(p[k + 5])

            if self.padding > 0:
                self.padding = self.padding
            else:
                self.padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape) // 2):
                x = (
                    (self.padding + mean_face_shape_x[i])
                    / (2 * self.padding + 1)
                    * self.face_size_side
                )
                y = (
                    (self.padding + mean_face_shape_y[i])
                    / (2 * self.padding + 1)
                    * self.face_size_side
                )
                to_points.append([x, y])
                from_points.append([shape[2 * i], shape[2 * i + 1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, _ = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

            from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
            to_center = [0, 0]
            to_center[1] = self.face_size_side * 0.4
            to_center[0] = self.face_size_side * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D(
                (from_center[0], from_center[1]), -1 * angle, scale
            )
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(
                img, rot_mat, (self.face_size_side, self.face_size_side)
            )
            crop_imgs.append(chips)

        return crop_imgs

    def __call__(self, batch: np.ndarray, landmarks: list) -> Tuple[list]:
        """
        Align faces on batch of images.

        Parameters
        ----------
        :param batch: type(numpy.ndarray) batch of images.
        :param landmarks: type(list) landmarks for each face on each image.
        :return: face_chips - list of cropped faces from each image, number_of_faces - list amount of faces on each image.
        """
        face_chips = []
        number_of_faces = []
        for i, element in enumerate(batch):
            chips = self.extract_image_chips(element, landmarks[i])
            number_of_faces.append(len(chips))
            if len(chips) != 0:
                chips = np.array(chips)
                face_chips.append(chips)

        return face_chips, number_of_faces
