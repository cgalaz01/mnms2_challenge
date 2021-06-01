from typing import List, Union, Tuple

from multiprocessing import Pool

import numpy as np
from scipy import ndimage

import SimpleITK as sitk


class Preprocess():
    
    @staticmethod
    def resample_image(image: sitk.Image, out_spacing: Tuple[float]=(1.0, 1.0, 1.0),
                       out_size: Union[None, Tuple[int]]=None, is_label: bool=False,
                       pad_value: float=0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
    
    
    @staticmethod
    def normalise_intensities(image: sitk.Image) -> sitk.Image:
        # Normalise image fro hypothetical 0-500 to 0-1 range
        normalised_image = sitk.Cast(image, sitk.sitkFloat32) / 500.0
        
        return normalised_image
    
    

class Registration():
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def _function_register(initial_transform, moving_image, fixed_image,
                           learning_rate, histogram_bins, sampling_rate,
                           seed) -> Tuple[sitk.Transform, float]:
    
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        registration_method = sitk.ImageRegistrationMethod()
            
        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogram_bins)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(sampling_rate, seed=seed)
        
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer settings.
        if learning_rate == None:
            estimate_learning_rate = registration_method.EachIteration
            learning_rate = 0
        else:
            estimate_learning_rate = registration_method.Never
            
        registration_method.SetOptimizerAsGradientDescent(learningRate=learning_rate,
                                                          numberOfIterations=100,
                                                          convergenceMinimumValue=1e-12,
                                                          convergenceWindowSize=10,
                                                          estimateLearningRate=estimate_learning_rate)
    
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        registration_method.SetInitialTransform(initial_transform, inPlace=True)        
        
        transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))
        
        
        return transform, registration_method.GetMetricValue()
        
    
    @staticmethod
    def _parallel_register(initial_transform, moving_image, fixed_image,
                           learning_rate_list, histogram_bins, sampling_rate,
                           seed) -> Tuple[sitk.Transform, float]:
        
        function_input = [(sitk.AffineTransform(initial_transform),
                           moving_image,
                           fixed_image,
                           learning_rate_list[i],
                           histogram_bins,
                           sampling_rate,
                           seed) for i in range(len(learning_rate_list))]
        
        with Pool() as pool:
            output_results = pool.starmap(Registration._function_register, function_input)

        selected_transform = None
        min_metric_value = 1e5
        for i in range(len(output_results)):
            transform, metric_value = output_results[i]
            if metric_value < min_metric_value:
                min_metric_value = metric_value
                selected_transform = transform
                
        return selected_transform, min_metric_value
        
    
    @staticmethod
    def _major_alignment(moving_image: sitk.Image, fixed_image: sitk.Image,
                         debug_output: int=0) -> Tuple[sitk.Transform, float]:
        debug_image_outputs = []
        debug_image_moving = []
        debug_image_fixed = []
        
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        
        initial_transform = sitk.AffineTransform(3)
        
        if debug_output > 0:
            debug_image = sitk.Resample(moving_image,
                                        fixed_image,
                                        initial_transform,
                                        sitk.sitkLinear,
                                        0.0,
                                        moving_image.GetPixelID())
            debug_image_moving.append(debug_image)
            debug_image_fixed.append(fixed_image)
        
        
        transform = initial_transform
        
        
        gaussian_sigma = [8, 4, 2, 1, 0]
        
        histogram_bins = 200
        learning_rate_list = [[8.0, 4.0, 2.0, 1.0, None],
                              [4.0, 2.0, 1.0, 0.5, None],
                              [2.0, 1.0, 0.5, 0.25, None],
                              [1.0, 0.5, 0.25, 0.1, None],
                              [0.5, 0.25, 0.1, 0.05, None]]
        sampling_rate = 1.0
        
        seed = 12453
        
        for i in range(len(gaussian_sigma)):
                
            numpy_fixed_image = sitk.GetArrayFromImage(fixed_image)    
            numpy_fixed_image = ndimage.gaussian_filter(numpy_fixed_image,
                                                        sigma=(1,
                                                               gaussian_sigma[i],
                                                               gaussian_sigma[i]),
                                                        mode='constant')
            
            
            tmp_fixed_image = sitk.GetImageFromArray(numpy_fixed_image)
            tmp_fixed_image.CopyInformation(fixed_image)
            
            numpy_moving_image = sitk.GetArrayFromImage(moving_image)
            numpy_moving_image = ndimage.gaussian_filter(numpy_moving_image,
                                                         sigma=(gaussian_sigma[i] / 2,
                                                                gaussian_sigma[i],
                                                                gaussian_sigma[i]),
                                                         mode='constant')
            
            tmp_moving_image = sitk.GetImageFromArray(numpy_moving_image)
            tmp_moving_image.CopyInformation(moving_image)
            
    
            transform, metric = Registration._parallel_register(sitk.AffineTransform(transform),
                                                                tmp_moving_image, tmp_fixed_image,
                                                                learning_rate_list[i], histogram_bins,
                                                                sampling_rate, seed)
        
            if debug_output > 0:
                debug_image = sitk.Resample(tmp_moving_image,
                                            tmp_fixed_image,
                                            transform,
                                            sitk.sitkLinear,
                                            0.0,
                                            tmp_moving_image.GetPixelID())
                debug_image_moving.append(debug_image)
                debug_image_fixed.append(tmp_fixed_image)
    
    
        
        final_transform = transform
        
        debug_image_outputs = [debug_image_moving, debug_image_fixed]
        
        if debug_output == 1:
            return final_transform, metric, debug_image_outputs
        else:
            return final_transform, metric
    
    
    @staticmethod
    def _minor_alignment(moving_image: sitk.Image, fixed_image: sitk.Image,
                         debug_output: int=0) -> Tuple[sitk.Transform, float]:
        debug_image_outputs = []
        debug_image_moving = []
        debug_image_fixed = []
        
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        
        initial_transform = sitk.AffineTransform(3)
        
        if debug_output > 0:
            debug_image = sitk.Resample(moving_image,
                                        fixed_image,
                                        initial_transform,
                                        sitk.sitkLinear,
                                        0.0,
                                        moving_image.GetPixelID())
            debug_image_moving.append(debug_image)
            debug_image_fixed.append(fixed_image)
        
        
        transform = initial_transform
        
        
        gaussian_sigma = [2, 1, 0]
        
        histogram_bins = 200
        learning_rate_list = [[2.0, 2.0, 1.0, 0.5, None],
                              [2.0, 1.0, 0.5, 0.25, None],
                              [1.0, 0.5, 0.25, 0.1, None],]
        sampling_rate = 1.0
        
        seed =  12453
        
        for i in range(len(gaussian_sigma)):
                
            numpy_fixed_image = sitk.GetArrayFromImage(fixed_image)    
            numpy_fixed_image = ndimage.gaussian_filter(numpy_fixed_image,
                                                        sigma=(1,
                                                               gaussian_sigma[i],
                                                               gaussian_sigma[i]),
                                                        mode='constant')
            
            
            tmp_fixed_image = sitk.GetImageFromArray(numpy_fixed_image)
            tmp_fixed_image.CopyInformation(fixed_image)
            
            numpy_moving_image = sitk.GetArrayFromImage(moving_image)
            numpy_moving_image = ndimage.gaussian_filter(numpy_moving_image,
                                                         sigma=(gaussian_sigma[i] / 2,
                                                                gaussian_sigma[i],
                                                                gaussian_sigma[i]),
                                                         mode='constant')
            
            tmp_moving_image = sitk.GetImageFromArray(numpy_moving_image)
            tmp_moving_image.CopyInformation(moving_image)
    
            transform, metric = Registration._parallel_register(sitk.AffineTransform(transform),
                                                                tmp_moving_image, tmp_fixed_image,
                                                                learning_rate_list[i], histogram_bins,
                                                                sampling_rate, seed)
        
            if debug_output > 0:
                debug_image = sitk.Resample(tmp_moving_image,
                                            tmp_fixed_image,
                                            transform,
                                            sitk.sitkLinear,
                                            0.0,
                                            tmp_moving_image.GetPixelID())
                debug_image_moving.append(debug_image)
                debug_image_fixed.append(tmp_fixed_image)
    
        
        final_transform = transform
        
        debug_image_outputs = [debug_image_moving, debug_image_fixed]
        
        if debug_output == 1:
            return final_transform, metric, debug_image_outputs
        else:
            return final_transform, metric
        
    
    @staticmethod
    def register(moving_image: sitk.Image, fixed_image: sitk.Image,
                 debug_output: int=0) -> Tuple[sitk.Transform, float, Union[None, List[List[sitk.Image]]]]:        
        major_output = Registration._major_alignment(moving_image, fixed_image, debug_output)
        minor_output = Registration._minor_alignment(moving_image, fixed_image, debug_output)

        if major_output[1] < minor_output[1]:
            return major_output
        else:
            return minor_output
        
    
    @staticmethod
    def get_affine_matrix(image: sitk.Image) -> np.ndarray:
        # get affine transform in LPS
        c = [image.TransformContinuousIndexToPhysicalPoint(p)
             for p in ((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (0, 0, 0))]
        c = np.array(c)
        affine = np.concatenate([
            np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
            [[0.], [0.], [0.], [1.]]
        ], axis=1)
        affine = np.transpose(affine)
        # convert to RAS to match nibabel etc.
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
        return affine
    
    
    @staticmethod
    def get_affine_registration_matrix(moving_image: sitk.Image,
                                       registration_affine: sitk.Transform) -> np.ndarray:
        # Get affine transform in LPS
        c = [registration_affine.TransformPoint(
                 moving_image.TransformContinuousIndexToPhysicalPoint(p))
             for p in ((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (0, 0, 0))]
        c = np.array(c)
        affine = np.concatenate([
            np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
            [[0.], [0.], [0.], [1.]]
        ], axis=1)
        affine = np.transpose(affine)
        # Convert to RAS to match nibabel etc.
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
        return affine
    
