import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from skimage.util.shape import view_as_windows
import time 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def view_as_window_conv(image, filter, stride = 1, padding= "VALID"):
    '''Conv operation with skimage's view_as_windows.

    image   : Input image array of (N X H X C).
    filter  : Filter array of size (nx, ny, nc).
    stride  : stride of windows, integer.

    Return  : convolved image output 
    '''
    m2,n2=filter.shape[:2]
    if padding != "SAME" and padding != "VALID":
        raise ValueError("Please choose a valid padding, options: 'SAME' or 'VALID'.") 

    if padding == "SAME":
        image = np.pad(image, pad_width=((m2//2,m2//2),(n2//2,n2//2),(0,0)))

    return np.tensordot(view_as_windows(image, filter.shape, step = stride), filter, axes=3)

def np_as_strided_conv(image, filter, stride = 1, padding = "VALID"):
    '''Conv operation with numpy's stride_tricks.

    image   : Input image array of NXHXC.
    filter  : Filter array of size: (ny, nx, nc).
    stride  : stride of windows, integer.

    Return  : convolved image output 
    '''
    if padding != "SAME" and padding != "VALID":
        raise ValueError("Please choose a valid padding, options: 'SAME' or 'VALID'.") 

    m2,n2=filter.shape[:2]

    if padding == "SAME":
        image = np.pad(image, pad_width=((m2//2,m2//2),(n2//2,n2//2),(0,0)))

    s0,s1=image.strides[:2]
    m1,n1=image.shape[:2]

    shape=(1+(m1-m2)//stride,1+(n1-n2)//stride,m2,n2)+image.shape[2:]
    strides=(stride*s0,stride*s1,s0,s1)+image.strides[2:]
    strided_array=np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)

    return np.tensordot(strided_array, filter, axes=3)

if __name__ == "__main__":
    image_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    np_conv_times = []
    scikit_conv_times = []
    tf_conv_times = []

    for padding_style in ["VALID", "SAME"]:
        for imagesize in image_sizes:
            image_array = np.random.rand(imagesize, imagesize, 3).astype("float32")
            sobel_y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]*3).reshape((3,3,3)).astype("float32")

            start_time = time.perf_counter()
            np_conv = np_as_strided_conv(image=image_array, filter=sobel_y_filter, stride=1, padding= padding_style)
            np_timing = time.perf_counter() - start_time
            print("Numpy stride tricks         --> image size: " + str(imagesize) + ", padding: " + str(padding_style) + ", time took :" + 
            str(np_timing) + ", shape :" + str(np_conv.shape) + ", conv output :" + str(np_conv[0:3,0])) 
            np_conv_times.append(np_timing)

            start_time = time.perf_counter()
            scikit_conv = view_as_window_conv(image=image_array, filter=sobel_y_filter, stride=1, padding=padding_style)
            scikit_timing = time.perf_counter() - start_time
            print("Scikit-Image strided window --> image size: " + str(imagesize) + ", padding: " + str(padding_style) + ", time took :" + 
            str(scikit_timing) + ", shape :" + str(np_conv.shape) + ", conv output :" + str(scikit_conv[0:3,0,0])) 
            scikit_conv_times.append(scikit_timing)

            start_time = time.perf_counter()
            tf_conv = tf.nn.conv2d(np.expand_dims(image_array, axis=0), np.expand_dims(sobel_y_filter, axis=3), strides=[1, 1, 1, 1], padding=padding_style)
            tf_timing = time.perf_counter() - start_time
            print("Tensorflow conv2D           --> image size: " + str(imagesize) + ", padding: " + str(padding_style) + ", time took :" + 
            str(tf_timing) + ", shape :" + str(np_conv.shape) + ", conv output :" + str(tf_conv[0,0:3,0,0]) + "\n")
            tf_conv_times.append(tf_timing)

    plt.figure(figsize=(18,8))
    plt.subplot(121)
    plt.plot(image_sizes, np_conv_times[:len(image_sizes)], label="numpy")
    plt.plot(image_sizes, scikit_conv_times[:len(image_sizes)], label="scikit")
    plt.plot(image_sizes, tf_conv_times[:len(image_sizes)], label="tensorflow")
    plt.xlabel("Image Size")
    plt.ylabel("Time Took in Seconds")
    plt.title("Performance Time for Conv2D, padding = 'VALID'")
    plt.legend()

    plt.subplot(122)
    plt.plot(image_sizes, np_conv_times[len(image_sizes):], label="numpy")
    plt.plot(image_sizes, scikit_conv_times[len(image_sizes):], label="scikit")
    plt.plot(image_sizes, tf_conv_times[len(image_sizes):], label="tensorflow")
    plt.xlabel("Image Size")
    plt.ylabel("Time Took in Seconds")
    plt.title("Performance Time for Conv2D, padding = 'SAME'")
    plt.legend()
    plt.savefig("comparision_plots.png", dpi=300)
    plt.show()


#### TERMINAL OUTPUT ####
# Numpy stride tricks         --> image size: 64, padding: VALID, time took :0.0005525999999997921, shape :(62, 62), conv output :[-0.10841548 -0.37044334  0.18744373]
# Scikit-Image strided window --> image size: 64, padding: VALID, time took :0.0005034000000003758, shape :(62, 62), conv output :[-0.10841548 -0.37044334  0.18744373]
# Tensorflow conv2D           --> image size: 64, padding: VALID, time took :0.19565310000000036, shape :(62, 62), conv output :tf.Tensor([-0.10841554 -0.37044334  0.1874435 ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 128, padding: VALID, time took :0.0014849999999997365, shape :(126, 126), conv output :[-3.0191207 -2.4963186 -1.2870512]
# Scikit-Image strided window --> image size: 128, padding: VALID, time took :0.0015890000000000626, shape :(126, 126), conv output :[-3.0191207 -2.4963186 -1.2870512]
# Tensorflow conv2D           --> image size: 128, padding: VALID, time took :0.0018476999999998966, shape :(126, 126), conv output :tf.Tensor([-3.019121  -2.4963186 -1.2870512], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 256, padding: VALID, time took :0.0054714999999996294, shape :(254, 254), conv output :[-2.326699  -1.7743351 -2.071836 ]
# Scikit-Image strided window --> image size: 256, padding: VALID, time took :0.006435699999999933, shape :(254, 254), conv output :[-2.326699  -1.7743351 -2.071836 ]
# Tensorflow conv2D           --> image size: 256, padding: VALID, time took :0.006184799999999768, shape :(254, 254), conv output :tf.Tensor([-2.3266988 -1.7743351 -2.071836 ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 512, padding: VALID, time took :0.027255400000000485, shape :(510, 510), conv output :[4.0697145 1.2489982 0.5997207]
# Scikit-Image strided window --> image size: 512, padding: VALID, time took :0.027153599999999223, shape :(510, 510), conv output :[4.0697145 1.2489982 0.5997207]
# Tensorflow conv2D           --> image size: 512, padding: VALID, time took :0.01864050000000006, shape :(510, 510), conv output :tf.Tensor([4.069715  1.2489982 0.5997211], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 1024, padding: VALID, time took :0.09624159999999993, shape :(1022, 1022), conv output :[ 0.12143493 -0.775084   -1.456781  ]
# Scikit-Image strided window --> image size: 1024, padding: VALID, time took :0.08549179999999978, shape :(1022, 1022), conv output :[ 0.12143493 -0.775084   -1.456781  ]
# Tensorflow conv2D           --> image size: 1024, padding: VALID, time took :0.032483500000000554, shape :(1022, 1022), conv output :tf.Tensor([ 0.12143499 -0.7750839  -1.456781  ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 2048, padding: VALID, time took :0.3345868000000003, shape :(2046, 2046), conv output :[-0.08032846 -1.0239761  -1.6161673 ]
# Scikit-Image strided window --> image size: 2048, padding: VALID, time took :0.3138683999999996, shape :(2046, 2046), conv output :[-0.08032846 -1.0239761  -1.6161673 ]
# Tensorflow conv2D           --> image size: 2048, padding: VALID, time took :0.15667710000000046, shape :(2046, 2046), conv output :tf.Tensor([-0.0803284 -1.0239763 -1.6161671], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 4096, padding: VALID, time took :1.1799248000000002, shape :(4094, 4094), conv output :[-3.088109   -0.31382442 -1.0059237 ]
# Scikit-Image strided window --> image size: 4096, padding: VALID, time took :1.3687526000000005, shape :(4094, 4094), conv output :[-3.088109   -0.31382442 -1.0059237 ]
# Tensorflow conv2D           --> image size: 4096, padding: VALID, time took :0.47762930000000026, shape :(4094, 4094), conv output :tf.Tensor([-3.0881093  -0.31382465 -1.0059239 ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 64, padding: SAME, time took :0.014068900000001605, shape :(64, 64), conv output :[-3.5508533 -3.8798537 -4.786222 ]
# Scikit-Image strided window --> image size: 64, padding: SAME, time took :0.01793319999999987, shape :(64, 64), conv output :[-3.5508533 -3.8798537 -4.786222 ]
# Tensorflow conv2D           --> image size: 64, padding: SAME, time took :0.01204899999999931, shape :(64, 64), conv output :tf.Tensor([-3.5508533 -3.8798537 -4.7862225], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 128, padding: SAME, time took :0.002113399999998933, shape :(128, 128), conv output :[-2.7491758 -5.678779  -6.9860325]
# Scikit-Image strided window --> image size: 128, padding: SAME, time took :0.00226729999999975, shape :(128, 128), conv output :[-2.7491758 -5.678779  -6.9860325]
# Tensorflow conv2D           --> image size: 128, padding: SAME, time took :0.0019244999999994405, shape :(128, 128), conv output :tf.Tensor([-2.7491758 -5.678779  -6.9860325], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 256, padding: SAME, time took :0.008200399999999775, shape :(256, 256), conv output :[-1.7501584 -4.276161  -6.1902666]
# Scikit-Image strided window --> image size: 256, padding: SAME, time took :0.014795899999999307, shape :(256, 256), conv output :[-1.7501584 -4.276161  -6.1902666]
# Tensorflow conv2D           --> image size: 256, padding: SAME, time took :0.006533200000001571, shape :(256, 256), conv output :tf.Tensor([-1.7501584 -4.276161  -6.190267 ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 512, padding: SAME, time took :0.0361659999999997, shape :(512, 512), conv output :[-5.720699  -7.5177813 -7.008318 ]
# Scikit-Image strided window --> image size: 512, padding: SAME, time took :0.04507560000000055, shape :(512, 512), conv output :[-5.720699  -7.5177813 -7.008318 ]
# Tensorflow conv2D           --> image size: 512, padding: SAME, time took :0.015117400000001169, shape :(512, 512), conv output :tf.Tensor([-5.7206993 -7.5177813 -7.008318 ], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 1024, padding: SAME, time took :0.1083393000000008, shape :(1024, 1024), conv output :[-4.5439596 -7.7178793 -6.24459  ]
# Scikit-Image strided window --> image size: 1024, padding: SAME, time took :0.10965260000000043, shape :(1024, 1024), conv output :[-4.5439596 -7.7178793 -6.24459  ]
# Tensorflow conv2D           --> image size: 1024, padding: SAME, time took :0.032604200000001526, shape :(1024, 1024), conv output :tf.Tensor([-4.54396 -7.71788 -6.24459], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 2048, padding: SAME, time took :0.33145060000000015, shape :(2048, 2048), conv output :[-4.930257  -7.698996  -6.7625275]
# Scikit-Image strided window --> image size: 2048, padding: SAME, time took :0.36170669999999916, shape :(2048, 2048), conv output :[-4.930257  -7.698996  -6.7625275]
# Tensorflow conv2D           --> image size: 2048, padding: SAME, time took :0.14469940000000037, shape :(2048, 2048), conv output :tf.Tensor([-4.9302564 -7.6989956 -6.7625275], shape=(3,), dtype=float32)

# Numpy stride tricks         --> image size: 4096, padding: SAME, time took :1.2892707999999988, shape :(4096, 4096), conv output :[-2.5453737 -4.440523  -4.5132084]
# Scikit-Image strided window --> image size: 4096, padding: SAME, time took :1.3288738999999996, shape :(4096, 4096), conv output :[-2.5453737 -4.440523  -4.5132084]
# Tensorflow conv2D           --> image size: 4096, padding: SAME, time took :0.5585503999999997, shape :(4096, 4096), conv output :tf.Tensor([-2.5453737 -4.440523  -4.5132084], shape=(3,), dtype=float32)
