from typing import Tuple, Union

import numpy as np

import SimpleITK as sitk


class DataAugmentation():
    
    def __init__(self, seed: Union[int, None]):
        self.random_generator = np.random.RandomState(seed)
        
        self.min_z_rotation_degrees = -180
        self.max_z_rotation_degrees = 180
    
        self.min_gaussian_blur_sigma = 0
        self.min_gaussian_blur_sigma = 6
        
    
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
    def _rotate_z_axis(image: sitk.Image, degrees: float) -> Tuple[sitk.Image, sitk.Euler3DTransform]:
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
        rotation_matrix = DataAugmentation.matrix_from_axis_angle(axis_angle)
        
        # Construct transfomration matrix
        transformation = sitk.Euler3DTransform()
        transformation.SetCenter(physical_centre)
        transformation.SetMatrix(rotation_matrix.flatten().tolist())
        
        rotated_image = sitk.Resample(image,
                                      transformation,
                                      sitk.sitkLinear,
                                      0)
        
        return rotated_image, transformation
    
    
    def _random_rotate_z_axis(self, image: sitk.Image, gt_image: Union[None, sitk.Image],
                              use_cache: bool) -> Tuple[sitk.Image, sitk.Euler3DTransform]:

        if use_cache and hasattr(DataAugmentation, '_cache_rotate_z_degrees'):
            degrees = self._cache_rotate_z_degrees
        else:
            degrees = self.random_generator.randint(self.min_z_rotation_degrees,
                                                    self.max_z_rotation_degrees,
                                                    size=None,
                                                    dtype=int)
            self._cache_rotate_z_degrees = degrees
        
        rotated_image, rotation_matrix = self._rotate_z_axis(image, degrees)
        
        if gt_image is not None:
            rotated_gt, rotation_matrix = self._rotate_z_axis(gt_image, degrees)
        
            return rotated_image, rotated_gt, rotation_matrix
        
        return rotated_image, rotation_matrix
    
    
    @staticmethod
    def _blur_image(image: sitk, gaussian_sigma: float) -> sitk.Image:
        # Store original pixel type to convert back to original as the smoothing
        # operation will convert it to float
        pixel_id = image.GetPixelID()

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(gaussian_sigma)
        image = gaussian.Execute(image)
        
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixel_id)
        image = caster.Execute(image)
     
        return image
    
    
    def _random_blur_image(self, image: sitk.Image, use_cache: bool = False) -> sitk.Image:
        if use_cache and hasattr(DataAugmentation, '_cache_blur_sigma'):
            sigma = self._cache_blur_sigma
        else:
            sigma = self.random_generator.uniform(self.min_gaussian_blur_sigma,
                                                  self.max_gaussian_blur_sigma)
            
            self._cache_blur_sigma = sigma
            
        image = self._blur_image(image, sigma)
        
        return image
    
    
    def random_augmentation(self, image: sitk.Image, gt_image: sitk.Image,
                            use_cache: bool = False) -> Tuple[sitk.Image, sitk.Image, sitk.Euler3DTransform]:
        
        image, gt_image, rotation = self._random_rotate_z_axis(image, gt_image, use_cache)
        image = self._random_blur_image(image, use_cache)
        
        return image, gt_image, rotation

