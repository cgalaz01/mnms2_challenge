from typing import List, Tuple, Union

import numpy as np
from scipy import ndimage

import SimpleITK as sitk


class DataAugmentation():
    
    def __init__(self, seed: Union[int, None]):
        self.random_generator = np.random.RandomState(seed)
        
        self.min_z_rotation_degrees = -30
        self.max_z_rotation_degrees = 30
    
        self.min_gaussian_blur_sigma = 0
        self.max_gaussian_blur_sigma = 2
        
        self.rayleigh_scale = 0.005
        
    
    @staticmethod
    def _matrix_from_axis_angle(a: Tuple[float]) -> np.ndarray:
        """ Compute rotation matrix from axis-angle.
        This is called exponential map or Rodrigues' formula.
        Parameters
        ----------
        a : array-like, shape (4,)
            Axis of rotation and rotation angle: (x, y, z, angle)
        Returns
        -------
        R : array-like, shape (3, 3)
            Rotation matrix
        """
        ux, uy, uz, theta = a
        c = np.cos(theta)
        s = np.sin(theta)
        ci = 1.0 - c
        R = np.array([[ci * ux * ux + c,
                       ci * ux * uy - uz * s,
                       ci * ux * uz + uy * s],
                      [ci * uy * ux + uz * s,
                       ci * uy * uy + c,
                       ci * uy * uz - ux * s],
                      [ci * uz * ux - uy * s,
                       ci * uz * uy + ux * s,
                       ci * uz * uz + c],
                      ])
    
        return R


    @staticmethod
    def _rotate_z_axis(image: sitk.Image, degrees: float, is_labels: bool) -> Tuple[sitk.Image, sitk.Euler3DTransform]:
        # Adapted from:
        #   https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
        
        radians = np.deg2rad(degrees)
        
        # Find image centre
        width, height, depth = image.GetSize()
        physical_centre = image.TransformIndexToPhysicalPoint((width // 2,
                                                               height // 2,
                                                               depth // 2))
        
        direction = image.GetDirection()
        axis_angle = (direction[2], direction[5], direction[8], radians)
        rotation_matrix = DataAugmentation._matrix_from_axis_angle(axis_angle)
        
        # Construct transfomration matrix
        transformation = sitk.Euler3DTransform()
        transformation.SetCenter(physical_centre)
        transformation.SetMatrix(rotation_matrix.flatten().tolist())
        
        if is_labels:
            interpolater = sitk.sitkNearestNeighbor
        else:
            interpolater = sitk.sitkLinear
        rotated_image = sitk.Resample(image,
                                      transformation,
                                      interpolater,
                                      0)
        
        return rotated_image, transformation
    
    
    def _random_rotate_z_axis(self, image: sitk.Image, gt_image: Union[None, sitk.Image],
                              use_cache: bool) -> Tuple[sitk.Image, Union[None, sitk.Image], sitk.Euler3DTransform]:

        if use_cache:
            degrees = self._cache_rotate_z_degrees
        else:
            degrees = self.random_generator.randint(self.min_z_rotation_degrees,
                                                    self.max_z_rotation_degrees,
                                                    size=None,
                                                    dtype=int)
            self._cache_rotate_z_degrees = degrees
        
        rotated_image, rotation_matrix = self._rotate_z_axis(image, degrees, is_labels=False)
        
        if gt_image is not None:
            rotated_gt, rotation_matrix = self._rotate_z_axis(gt_image, degrees, is_labels=True)
        
            return rotated_image, rotated_gt, rotation_matrix
        
        return rotated_image, rotation_matrix

        
    @staticmethod
    def _blur_image(image: sitk.Image, gaussian_sigma: float) -> sitk.Image:
        numpy_image = sitk.GetArrayFromImage(image)
        
        # In-plane only blurring
        numpy_image = ndimage.gaussian_filter(numpy_image, (gaussian_sigma,
                                                            gaussian_sigma,
                                                            0))
        
        blurred_image = sitk.GetImageFromArray(numpy_image)
        blurred_image.CopyInformation(image)
        
        return blurred_image
    
    
    def _random_blur_image(self, image: sitk.Image, use_cache: bool = False) -> sitk.Image:
        if use_cache:
            sigma = self._cache_blur_sigma
        else:
            sigma = self.random_generator.uniform(self.min_gaussian_blur_sigma,
                                                  self.max_gaussian_blur_sigma)
            
            self._cache_blur_sigma = sigma
            
        image = self._blur_image(image, sigma)
        
        return image
    
    
    def _random_noise(self, image: sitk.Image) -> sitk.Image:
        numpy_image = sitk.GetArrayFromImage(image)
        rayleigh_noise = self.random_generator.rayleigh(self.rayleigh_scale,
                                                        size=numpy_image.shape)
        negative_noise = 2 * self.random_generator.randint(0, 2, size=numpy_image.shape) - 1
        numpy_image += (rayleigh_noise * negative_noise)
        
        noisy_image = sitk.GetImageFromArray(numpy_image)
        noisy_image.CopyInformation(image)
        
        return noisy_image
    
    
    def random_augmentation(self, image: sitk.Image, gt_image: sitk.Image,
                            use_cache: bool = False) -> Tuple[sitk.Image, sitk.Image, sitk.Euler3DTransform]:
        image, gt_image, rotation = self._random_rotate_z_axis(image, gt_image, use_cache)
        image = self._random_blur_image(image, use_cache)
        image = self._random_noise(image)
        
        return image, gt_image, rotation

