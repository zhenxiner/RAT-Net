import numpy as np
import torch
import torch.nn.functional as F
# from hausdorff_cal.hausdorff import hausdorff_distance
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure

# smooth = 0
smooth = 1e-5
def fast_hist(target, output):
    n_class = 2
    mask = (target >= 0) & (target < n_class)
    hist = torch.bincount(
        n_class * target[mask].int() + output[mask].int(),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def Dice(hist, output, target):
    if target.sum() == 0:
        return 1

    else:
        intersection = 2 * hist[1][1]
        return (intersection + smooth) / (output.sum() + target.sum() + smooth)

def Miou(hist):
    return (hist[1][1] + smooth) / (hist[1][1] + hist[1][0] +hist[0][1] + smooth)

def Sensitivity(hist, target):
    intersection = hist[1][1]
    return (intersection + smooth) / (target.sum() + smooth)

def Specificity(hist, target):
    intersection = hist[0][0]
    return (intersection) / (hist.sum() - target.sum() + smooth)

def ACC(hist, target):
    intersection = hist[0][0] + hist[1][1]
    return (intersection) / (hist.sum())


def surface_distances(result, reference,  connectivity=1):
    result = np.atleast_1d(result.astype(np.bool))# 0 1转化为boolean图
    reference = np.atleast_1d(reference.astype(np.bool))

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    # if 0 == np.count_nonzero(result):
    #     raise RuntimeError('The first supplied array does not contain any binary object.')
    # if 0 == np.count_nonzero(reference):
    #     raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=None)
    sds = dt[result_border]

    return sds

#    95th percentile of the Hausdorff Distance.
def Hd95(result, reference,  connectivity=1):
    if torch.is_tensor(result):
        result = result.data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()

    hd1 = surface_distances(result, reference, connectivity)
    hd2 = surface_distances(reference, result,  connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95