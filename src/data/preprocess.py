from typing import List, Union, Tuple

from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from skimage import color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from sklearn.cluster import KMeans

import SimpleITK as sitk


class RegionOfInterest():
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def _mean_roi_centroid(centroids_x: Tuple[int], centroids_y: Tuple[int]) -> Tuple[int]:
        centroid_x = int(np.asarray(centroids_x).mean())    
        centroid_y = int(np.asarray(centroids_y).mean())
        
        return centroid_x, centroid_y
    
    
    @staticmethod
    def detect_roi_sa(sitk_ed_image: sitk.Image, sitk_es_image: sitk.Image,
                      debug: bool = False) -> Tuple[int]:
        ed_image = sitk.GetArrayFromImage(sitk_ed_image)
        es_image = sitk.GetArrayFromImage(sitk_es_image)
        ed_image = np.swapaxes(ed_image, 0, -1)
        es_image = np.swapaxes(es_image, 0, -1)
        
        if sitk_ed_image.GetSize()[-1] > 4:
            z_indexes = [0, 2, 4]
        else:
            z_indexes = [0, 2]
        
        all_cx = []
        all_cy = []
        for z in z_indexes:
            ed_slice = ed_image[:, :, z]
            es_slice = es_image[:, :, z]
            
            width = ed_slice.shape[0]
            height = ed_slice.shape[1]
            image_size = (width + height) // 2
                
            diff_image = abs(ed_slice - es_slice)
            edge_image = canny(diff_image, sigma=2.0, low_threshold=0.8, high_threshold=0.98,
                               use_quantiles=True)
            
            lower_range = int(image_size * 0.08)    
            upper_range = int(image_size * 0.28)
            hough_radii = np.arange(lower_range, upper_range, 3)
            hough_res = hough_circle(edge_image, hough_radii)
            
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=10,
                                                       normalize=False)
            
            all_cx.extend(cx)
            all_cy.extend(cy)
            
            if debug:
                import matplotlib.pyplot as plt
                
                plt.imshow(ed_slice)
                plt.show()
                plt.close()
                plt.imshow(es_slice)
                plt.show()
                plt.close()
                
                plt.imshow(diff_image)
                plt.show()
                plt.close()
                 
                plt.imshow(edge_image)
                plt.show()
                plt.close()
                
                mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(cx, cy)
                
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
                image = ((ed_slice - ed_slice.min()) *
                         (1 / (ed_slice.max() - ed_slice.min()) * 255)).astype('uint8')
                image = color.gray2rgb(image)
                for center_y, center_x, radius in zip(cy, cx, radii):
                    circy, circx = circle_perimeter(center_y, center_x, radius,
                                                    shape=image.shape)
                    image[circy, circx] = (220, 20, 20)
                
                ax.imshow(image, cmap=plt.cm.gray)
                ax.scatter([mean_cx], [mean_cy])
                plt.show()
            
        mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(all_cx, all_cy)
        
        return mean_cx, mean_cy

    
    @staticmethod
    def detect_roi_la(sitk_ed_image: sitk.Image, sitk_es_image: sitk.Image,
                      debug: bool = False) -> Tuple[int]:
        ed_image = sitk.GetArrayFromImage(sitk_ed_image)
        es_image = sitk.GetArrayFromImage(sitk_es_image)
        ed_image = np.swapaxes(ed_image, 0, -1)
        es_image = np.swapaxes(es_image, 0, -1)
        
        ed_slice = ed_image[:, :, 0]
        es_slice = es_image[:, :, 0]
        
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.6, high_threshold=0.96,
                           use_quantiles=True)
        
        width = ed_slice.shape[1]
        height = ed_slice.shape[0]
        image_size = (width + height) // 2
        
        lower_range = int(image_size * 0.1)    
        upper_range = int(image_size * 0.35)
        hough_radii = np.arange(lower_range, upper_range, 3)
        hough_res = hough_circle(edge_image, hough_radii)
        
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=10,
                                                   normalize=False)
        
        mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(cx, cy)
        
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(ed_slice)
            plt.show()
            plt.close()
            plt.imshow(es_slice)
            plt.show()
            plt.close()
            
            plt.imshow(diff_image)
            plt.show()
            plt.close()
            
            plt.imshow(edge_image)
            plt.show()
            plt.close()
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            image = ((ed_slice - ed_slice.min()) *
                     (1/(ed_slice.max() - ed_slice.min()) * 255)).astype('uint8')
            image = color.gray2rgb(image)
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=image.shape)
                image[circy, circx] = (220, 20, 20)
            
            ax.imshow(image, cmap=plt.cm.gray)
            ax.scatter([mean_cx], [mean_cy])
            plt.show()
         
        
        return mean_cx, mean_cy



class Preprocess():
    
    @staticmethod
    def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (1.0, 1.0, 1.0),
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
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
    def _get_bounds_and_padding(dimension_image_size: int, dimension_centroid: int,
                                dimension_crop_length: int) -> Tuple[int]:
        
        min_index = dimension_centroid - dimension_crop_length // 2
        pad_min_value = 0
        if min_index < 0:
            pad_min_value = abs(min_index)
            min_index = 0
        
        max_index = min_index + dimension_crop_length
        pad_max_value = 0
        if max_index > dimension_image_size:
            pad_max_value = max_index - dimension_image_size
    
        return min_index, max_index, pad_min_value, pad_max_value
    
    
    @staticmethod
    def crop(image: sitk.Image, centroid: Tuple[int], length: Tuple[int],
             ignore_z_axis: bool = False, padding: float = 0) -> sitk.Image:
        size = image.GetSize()
        
        min_x, max_x, pad_min_x, pad_max_x = Preprocess._get_bounds_and_padding(size[0],
                                                                                centroid[0],
                                                                                length[0])
        
        min_y, max_y, pad_min_y, pad_max_y = Preprocess._get_bounds_and_padding(size[1],
                                                                                centroid[1],
                                                                                length[1])
        
        min_z, max_z, pad_min_z, pad_max_z = Preprocess._get_bounds_and_padding(size[2],
                                                                                centroid[2],
                                                                                length[2])

        if ignore_z_axis:
            pad_min_z = 0
            pad_max_z = 0
            min_z = 0
            max_z = size[-1]
        
        lower_padding = np.asarray([pad_max_x, pad_max_y, pad_max_z]).astype(np.uint32).tolist()
        upper_padding = np.asarray([pad_min_x, pad_min_y, pad_min_z]).astype(np.uint32).tolist()
        padded_image = sitk.ConstantPad(image, upper_padding, lower_padding, padding)
        
        cropped_image = padded_image[min_x: max_x, min_y: max_y, min_z: max_z]
        
        return cropped_image
    
    
    @staticmethod
    def pad(image: sitk.Image, lower_bound: Tuple[int] = [0, 0, 0],
            upper_bound: Tuple[int] = [0, 0, 0], constant: float = 0) -> sitk.Image:
        pad_filter = sitk.ConstantPadImageFilter()

        pad_filter.SetConstant(constant)
        pad_filter.SetPadLowerBound(lower_bound)
        pad_filter.SetPadUpperBound(upper_bound)
        padded_image = pad_filter.Execute(image)
        
        return padded_image
    
    
    @staticmethod
    def normalise_intensities(image: sitk.Image) -> sitk.Image:
        # Normalise image to 0-1 range
        numpy_image = sitk.GetArrayViewFromImage(image)
        normalised_image = sitk.Cast(image, sitk.sitkFloat32) / numpy_image.max()
        
        return normalised_image
    
    
    @staticmethod
    def z_score_normalisation(image: sitk.Image) -> sitk.Image:
        epsilon = 1e-7
        numpy_image = sitk.GetArrayViewFromImage(image)
        mean_value = numpy_image.mean()
        standard_deviation = numpy_image.std()
        
        normalised_image = ((sitk.Cast(image, sitk.sitkFloat32) - mean_value) /
                            (standard_deviation + epsilon))
        
        return normalised_image
    

    @staticmethod
    def z_score_patch_normalisation(image_ed: sitk.Image, image_es: sitk.Image,
                                    image_type: str) -> Tuple[sitk.Image]:
        epsilon = 1e-7
        if image_type == 'sa':    
            patch_x = 21
            patch_y = 21
        else:
            patch_x = 31
            patch_y = 31
        
        numpy_image = sitk.GetArrayFromImage(image_ed)
        numpy_image = np.swapaxes(numpy_image, 0, -1)
        
        # Assumes the heart is alligned to the centre of the image
        centre_x = image_ed.GetSize()[0] // 2
        centre_y = image_ed.GetSize()[1] // 2
        min_x = centre_x - patch_x // 2
        max_x = min_x + patch_x
        
        min_y = centre_y - patch_y // 2
        max_y = min_y + patch_y
        
        if image_ed.GetSize()[2] > 4:
            centre_z = 2
        else:
            centre_z = 0
        
        roi_patch = numpy_image[min_x: max_x+1, min_y: max_y+1, centre_z]
        roi_patch = roi_patch.reshape(-1, 1)
        roi_patch = np.sort(roi_patch)
        
        centroids = np.asarray([[0], [len(roi_patch) - 1]])
        cluster_labels = KMeans(n_clusters=2, init=centroids,
                                n_init=1, random_state=0).fit_predict(roi_patch)
        cluster_1 = roi_patch[cluster_labels == 0]
        cluster_2 = roi_patch[cluster_labels == 1]
        
        cluster_1_mean = cluster_1.mean()
        cluster_2_mean = cluster_2.mean()
        
        if cluster_1_mean > cluster_2_mean:
            mean_value = cluster_1_mean
            standard_deviation = cluster_1.std()
        else:
            mean_value = cluster_2_mean
            standard_deviation = cluster_2.std()
            
        normalised_image_ed = ((sitk.Cast(image_ed, sitk.sitkFloat32) - mean_value) /
                               (standard_deviation + epsilon))
        
        normalised_image_es = ((sitk.Cast(image_es, sitk.sitkFloat32) - mean_value) /
                               (standard_deviation + epsilon))
        
        return normalised_image_ed, normalised_image_es
    
    
        
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
                         debug_output: int = 0) -> Tuple[sitk.Transform, float]:
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
                         debug_output: int = 0) -> Tuple[sitk.Transform, float]:
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
                              [1.0, 0.5, 0.25, 0.1, None]]
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
    def _fast_alignment(moving_image: sitk.Image, fixed_image: sitk.Image,
                        debug_output: int = 0) -> Tuple[sitk.Transform, float]:
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
        
        gaussian_sigma = [2, 0]
        histogram_bins = 40
        learning_rate_list = [[2.0, 1.0, 0.5, None],
                              [1.0, 0.5, 0.1, None]]
        sampling_rate = 1.0
        seed =  12453
        
        for i in range(len(gaussian_sigma)):
            if gaussian_sigma[i] > 0:
                numpy_fixed_image = sitk.GetArrayFromImage(fixed_image)    
                numpy_fixed_image = ndimage.gaussian_filter(numpy_fixed_image,
                                                            sigma=(0,
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
            else:
                tmp_fixed_image = fixed_image
                tmp_moving_image = moving_image
    
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
                 fast_register: bool = True, debug_output: int = 0) -> Tuple[sitk.Transform, float, Union[None, List[List[sitk.Image]]]]:        
        if fast_register:
            fast_output = Registration._fast_alignment(moving_image, fixed_image, debug_output)
            return fast_output
        else:   
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
    
