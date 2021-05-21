import uuid
import numpy as np
from termcolor import cprint
cimport numpy as np
from PIL import Image
from libc.math cimport exp


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

ctypedef np.float32_t NLMIMGTYPE        


def _throw_or_return(bint result, bint throw, error=ValueError):
    if throw and not result:
        raise error()
    return result


def is_gt(*args, float threshold=0, bint throw=True):
    result = all([test > threshold for test in args])
    return _throw_or_return(result, throw)


def is_lt(*args, float threshold=0, bint throw=True):
    result = all([test < threshold for test in args])
    return _throw_or_return(result, throw)


def is_int(*args, bint throw=True):
    result = all([isinstance(test, int) for test in args])
    return _throw_or_return(result, throw)



cdef class ImageModel:
    cdef public np.ndarray pixels

    def __init__(self, *args, **kwargs):
        image = kwargs.get('image') or args[0]
        if not image:
            raise AttributeError('No image supplied to ModelImage instance.')
        image = Image.open(image).convert('L')
        #image = Image.open(image).convert('L')
        
         #('i:', <PIL.Image.Image image mode=RGB size=774x518 at 0x1252BCDF0>) - RGB
         #('i:', <PIL.Image.Image image mode=L size=774x518 at 0x1312DCDF0>) - L
        self.pixels = ImageModel.pixel_matrix(image.getdata())

    @staticmethod
    def pixel_matrix(image):
        cdef int rows, cols
        rows = image.size[1]
        cols = image.size[0]

        matrix = np.array(image)
        #print(matrix)
        #print(np.array(image.convert('RGB')))
        matrix.shape = rows, cols
        return matrix

    @staticmethod
    def setup():
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save(self, image):
        cdef str uid = str(uuid.uuid4())[:5]
        cdef str ext = 'jpg'
        cdef str name = self.__class__.__name__
        image.save('img/out/{name}-{hash}.{ext}'.format(
            name=name, hash=uid, ext=ext))
        return uid


# NLM implementation.
cdef class NLM(ImageModel):
    @staticmethod
    def setup(arr):
        cdef NLM model
        cdef str default_image = 'alleyNoisy_sigma20.png'
        try:
            #image_name = input('Enter the image name (default: {}).\n'
            #                   ' -> img/in/'.format(default_image))
            image_name = arr[0]
            if not image_name:
                cprint('Using default image: {}'
                       .format(default_image), 'yellow')
            #patch_radius = int(input('Enter the patch radius.\n -> '))
            patch_radius = int(arr[1])
            window_radius = int(arr[2])
            sigma = int(arr[3])
            #window_radius = int(input('Enter the window radius.\n -> '))
            #sigma = int(input('Enter the sigma value.\n -> '))
            _ = is_int(patch_radius, window_radius, sigma)
            _ = is_gt(patch_radius, window_radius, sigma, threshold=0)
            return NLM(
                'img/in/{}'.format(image_name or default_image),
                sigma=sigma,
                patch_radius=patch_radius,
                window_radius=window_radius,)
        except ValueError:
            print('The image name must be a valid string.'
                  'Other parameters must be positive integers.')
            return NLM.setup()

    cdef public int d, ds, D, Ds, sigma, distance_threshold
    cdef public float h

    def __init__(self, image, sigma, patch_radius, window_radius):
        super().__init__(image)
        self.d = 2 * patch_radius + 1
        self.ds = patch_radius
        self.D = 2 * window_radius + 1
        self.Ds = window_radius
        self.distance_threshold = 300
        self.h = .1
        self.sigma = sigma

    @staticmethod
    cdef pad_image(np.ndarray image, int pad_size, str mode='reflect'):
        return np.pad(image, pad_size, mode=mode)

    @staticmethod
    cdef integral_image(
            NLMIMGTYPE [:, :] padded_im,
            NLMIMGTYPE [:, ::] integral,
            int t_row,
            int t_col,
            int num_rows,
            int num_cols):
        cdef int x1, x2
        cdef float d
        for x1 in range(int_max(1, -t_row),
                        int_min(num_rows, num_rows - t_row)):
            for x2 in range(int_max(1, -t_col),
                            int_min(num_cols, num_cols - t_col)):
                d = (padded_im[x1, x2] - padded_im[x1 + t_row, x2 + t_col]) ** 2
                integral[x1, x2] = d \
                    + integral[x1 - 1, x2] \
                    + integral[x1, x2 - 1] \
                    - integral[x1 - 1, x2 - 1]

    @staticmethod
    cdef dist_from_integral_image(
        NLMIMGTYPE[:, ::] integral, int row, int col, int ds):
        cdef float d
        d = integral[row + ds, col + ds] \
            - integral[row + ds, col - ds - 1] \
            - integral[row - ds - 1, col + ds] \
            + integral[row - ds - 1, col - ds - 1]

        return d / (2 * ds + 1) ** 2

    def run(self, *args, **kwargs):
        cdef int num_rows, num_cols, pad_size, shift_row, shift_col, x1, x2, row, col
        cdef float double_coefficient, d2, weight

        pad_size = self.ds + self.Ds

        cdef NLMIMGTYPE [:, :] padded_im = np.ascontiguousarray(
            NLM.pad_image(self.pixels, pad_size)).astype(np.float32)
        cdef NLMIMGTYPE [:, :] output = np.zeros_like(padded_im)
        cdef NLMIMGTYPE [:, ::] weights = np.zeros_like(
            padded_im, dtype=np.float32, order='C')
        cdef NLMIMGTYPE[:, ::] integral_diff = np.zeros_like(
            padded_im, order='C')

        num_rows, num_cols = padded_im.shape[0], padded_im.shape[1]


        for shift_row in range(-self.Ds, self.Ds + 1):
            print('{}/{}'.format(shift_row + self.Ds, self.Ds + self.Ds))
            for shift_col in range(0, self.Ds + 1):
                double_coefficient = .5 if shift_row != 0 and shift_col == 0 else 1
                integral_diff = np.zeros_like(padded_im, order='C')
                NLM.integral_image(
                    padded_im, integral_diff, shift_row, shift_col, num_rows, num_cols)

                for x1 in range(int_max(self.ds, self.ds - shift_row),
                                int_min(num_rows - self.ds,
                                    num_rows - self.ds - shift_row)):
                    for x2 in range(int_max(self.ds, self.ds - shift_col),
                                    int_min(num_cols - self.ds,
                                        num_cols - self.ds - shift_col)):

                        d2 = NLM.dist_from_integral_image(
                            integral_diff, x1, x2, self.ds)

                        if d2 > self.distance_threshold:
                            continue
                        weight = double_coefficient * exp(
                            -(max(d2 - (2 * self.sigma ** 2), 0))
                            / self.h ** 2)


                        weights[x1, x2] += weight
                        weights[x1 + shift_row, x2 + shift_col] += weight

                        output[x1, x2] += weight * \
                            padded_im[x1 + shift_row, x2 + shift_col]
                        output[x1 + shift_row, x2 + shift_col] += weight * \
                            padded_im[x1, x2]


        for row in range(self.ds, num_rows - self.ds):
            for col in range(self.ds, num_cols - self.ds):
                output[row, col] /= weights[row, col]

        cdef np.ndarray normalised_output = np.array(
            output[pad_size:-pad_size, pad_size:-pad_size], dtype=np.uint8)

        cdef str im_id = self.save(Image.fromarray(normalised_output))

        print('Done! Image ID {}'.format(im_id))
        return normalised_output
