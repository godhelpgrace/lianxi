import cv2
import matplotlib.pyplot as plt
import numpy
from PIL import Image

"""
`image[:,:,0]`表示图像的B通道
`image[:,:,1]`表示图像的G通道
`image[:,:,2]`表示图像的R通道
`image[:,:,3]`表示图像的alpha通道

`cv2.imread()`函数读取图像时，使用BGR格式   接受一个参数：图像路径
`cv2.IMREAD_COLOR`表示读取彩色图像
`cv2.IMREAD_GRAYSCALE`表示读取灰度图像
`cv2.IMREAD_UNCHANGED`表示读取图像的alpha通道

`cv2.imshow()`函数显示图像时，使用BGR格式	接受两个参数：窗口名称和图像矩阵
`cv2.imwrite()`函数保存图像时，使用BGR格式	接受两个参数：保存路径和图像矩阵
`cv2.cvtColor()`函数接受两个参数：要转换的图像矩阵和转换的颜色空间
`cv2.COLOR_RGB2BGR`表示将RGB格式的图像转换为BGR格式
`cv2.COLOR_RGB2GRAY`表示将RGB格式的图像转换为灰度图像
`cv2.COLOR_RGB2HSV`表示将RGB格式的图像转换为HSV格式
`cv2.COLOR_RGB2YCrCb`表示将RGB格式的图像转换为YCrCb格式
`cv2.COLOR_BGR2RGB`表示将BGR格式的图像转换为RGB格式

`plt.imsave()`函数保存图像时，使用RGB格式
`PIL.Image.open()`函数读取图像时，使用RGB格式
`PIL.Image.save()`函数保存图像时，使用RGB格式
`skimage.io.imread()`函数读取图像时，使用RGB格式
`skimage.io.imsave()`函数保存图像时，使用RGB格式

`cv2.waitKey()`函数接受一个参数：等待时间
`cv2.waitKey(0)`表示无限等待
`cv2.destroyAllWindows()`函数用于销毁所有窗口

`skimage.color.rgb2gray()`函数将RGB格式的图像转换为灰度图像时，使用RGB格式
`skimage.color.gray2rgb()`函数将灰度图像转换为RGB格式的图像时，使用RGB格式
`skimage.color.rgb2hsv()`函数将RGB格式的图像转换为HSV格式的图像时，使用RGB格式
`skimage.color.hsv2rgb()`函数将HSV格式的图像转换为RGB格式的图像时，使用RGB格式
`skimage.color.rgb2ycbcr()`函数将RGB格式的图像转换为YCbCr格式的图像时，使用RGB格式
`skimage.color.ycbcr2rgb()`函数将YCbCr格式的图像转换为RGB格式的图像时，使用RGB格式
"""


def bgr2rgb(image):
    """
    BGR转RGB
    @param image: BGR图像矩阵
    @return:
    """
    # 当使用OpenCV库读取图像时，默认情况下，读入的图像通道顺序是BGR（蓝绿红）而不是RGB（红绿蓝）。
    # 这是由于在OpenCV中，图像是以BGR通道顺序存储的。因此，如果您想在OpenCV中使用RGB顺序，请在读取图像时进行通道顺序转换。
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 方法2
    # return image[:, :, (2, 1, 0)]
    # # 方法3
    # return image[...,::-1]


def rgb2gray(image):
    """
    彩色图像转换为灰度图像
    @param image: RGB图像矩阵
    @return:
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# RGB转灰度图像，原始方法
def rgb2gray_origin(image):
    """
    RGB转灰度图像，原始方法
    @param image: RGB图像矩阵
    @return:
    """
    # `cv2.split()`函数接受一个参数：要分割的图像
    # `cv2.split()`函数返回一个列表，列表中的每个元素是图像的一个通道
    r, g, b = cv2.split(image)
    # `cv2.merge()`函数接受一个参数：要合并的图像
    # `cv2.merge()`函数返回一个图像
    return cv2.merge([b * 0.11, g * 0.59, r * 0.3])

# RGB转灰度图像，直接计算
def rgb2gray_calc(image):
    """
    RGB转灰度图像，直接计算
    @param image: RGB图像矩阵
    @return:
    """
    return image[:, :, 0] * 0.11 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.3

def rgb2gray_loop(image):
    """
    RGB转灰度图像，遍历图片所有像素点
    @param image: RGB图像矩阵
    @return:
    """
    # `image.shape`返回一个元组，元组中的三个元素分别表示图像的高度、宽度和通道数
    height, width, _ = image.shape
    # `numpy.zeros()`函数接受一个参数：图像的大小
    # `numpy.zeros()`函数返回一个图像
    gray = numpy.zeros((height, width), numpy.uint8)    # 创建一张和当前图片大小一样的单通道图片
    # 遍历图片所有像素点
    for i in range(height):
        for j in range(width):
            gray[i, j] = int(image[i, j, 0] * 0.11 + image[i, j, 1] * 0.59 + image[i, j, 2] * 0.3)  # 将当前像素点的RGB值转换为灰度值
    return gray

def rgb2gray_loop_numpy(image):
    """
    RGB转灰度图像，遍历图片所有像素点，使用numpy
    @param image: RGB图像矩阵
    @return:
    """
    # `image.shape`返回一个元组，元组中的三个元素分别表示图像的高度、宽度和通道数
    height, width, _ = image.shape
    # `numpy.zeros()`函数接受一个参数：图像的大小
    # `numpy.zeros()`函数返回一个图像
    gray = numpy.zeros((height, width), numpy.uint8)
    for i in range(height):
        for j in range(width):
            gray[i, j] = numpy.dot(image[i, j, :], [0.11, 0.59, 0.3])   # 将当前像素点的RGB值转换为灰度值
    return gray

# RGB转灰度图像，使用Image模块
def rgb2gray_image(image):
    """
    RGB转灰度图像，使用Image模块
    @param image: RGB图像矩阵
    @return:
    """
    # `Image.fromarray()`函数接受一个参数：图像矩阵
    # `Image.fromarray()`函数返回一个图像
    image = Image.fromarray(image)
    # image = Image.open(image)     # 也可以使用`Image.open()`函数
    # `image.convert()`函数接受一个参数：图像的模式
    # `image.convert()`函数返回一个图像
    return image.convert('L')

def bgr2gray(image):
    """
    彩色图像转换为灰度图像
    @param image: BGR图像矩阵
    @return:
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 彩色图像转换为灰度图像,平均值法
def bgr2gray_avg(image):
    """
    彩色图像转换为灰度图像,平均值法
    @param image: BGR图像矩阵
    @return:
    """
    # `cv2.split()`函数接受一个参数：要分割的图像
    # `cv2.split()`函数返回一个列表，列表中的每个元素是图像的一个通道
    b, g, r = cv2.split(image)
    # b, g, r = cv2.mean(image)[:3]

    # `cv2.addWeighted()`函数接受四个参数：图像1、图像1的权重、图像2、图像2的权重
    # `cv2.addWeighted()`函数返回一个图像，该图像是图像1和图像2的加权和
    return cv2.addWeighted(b, 1/3, g, 1/3, 0) + cv2.addWeighted(r, 1/3, 0, 0, 0)

# 彩色图像转换为灰度图像,平均值法，使用numpy
def bgr2gray_avg_numpy(image):
    """
    彩色图像转换为灰度图像,平均值法，使用numpy
    @param image: BGR图像矩阵
    @return:
    """
    # `numpy.mean()`函数接受两个参数：图像矩阵和轴
    # `numpy.mean()`函数返回图像矩阵沿着指定轴的均值
    # `axis=2`表示沿着图像的高度和宽度计算均值，即取每个像素点的三个通道的平均值
    return numpy.mean(image, axis=2)


def read_image(image_path):
    """
    读取图像
    @param image_path: 图像路径
    @return:
    """
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def save_image(image, image_path):
    """
    保存图像
    @param image: 图像矩阵
    @param image_path: 图像路径
    @return:
    """
    # `cv2.imwrite()`函数接受两个参数：图像路径和图像矩阵
    cv2.imwrite(image_path, image)

def show_image(image):
    """
    显示图像
    @param image: 图像矩阵
    @return:
    """
    # `cv2.imshow()`函数接受两个参数：窗口名称和图像矩阵
    cv2.imshow('Image', image)
    # `cv2.waitKey()`函数接受一个参数：等待时间
    # `0`表示无限等待
    cv2.waitKey(0)
    # `cv2.destroyAllWindows()`函数用于销毁所有窗口
    cv2.destroyAllWindows()

def binaryzation(image, thresh=127, maxval=255):
    """
    二值化
    @param image: 灰度图像矩阵
    @param thresh: 阈值
    @param maxval: 最大值
    @return:
    """
    # `cv2.threshold()`函数接受三个参数：图像矩阵、阈值和最大值
    # `cv2.THRESH_BINARY`表示二值化
    # `ret`表示阈值
    # `binary`表示二值化后的图像矩阵

    # 判断图像是否为灰度图像
    if len(image.shape) == 3:
        # OPENCV读取的图像是BGR格式的，所以需要将BGR格式的图像转换为灰度图像
        image = bgr2gray(image)
    ret, binary = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
    return binary

def is_rgb(image):
    """
    判断图像是BGR格式，还是RGB格式, 存在误判的可能性
    @param image: 图像矩阵
    @return: True表示RGB格式，False表示BGR格式
    """
    # 获取图像的形状和通道数
    # height, width, channels = image.shape
    # 如果图像的形状是(height, width, 3)，那么这张图像就是RGB格式的。
    # 如果图像的形状是(height, width, 4)，那么这张图像就是RGBA格式的。
    # 如果图像的形状是(height, width)，那么这张图像就是灰度图像。
    if len(image.shape) == 3:   # 判断图像是否是三通道图像
        if image.shape[2] == 3: # 判断图像的通道数是否是3,如果是3，那么就是RGB格式
            return True
    return False

def main():
    # 读取图像
    image = read_image('lenna.png')
    # 显示图像
    show_image(image)

    # 将BGR格式的图像转换为RGB格式
    rgb_image = bgr2rgb(image)
    # 显示转换后的RGB格式的图像
    show_image(rgb_image)

    # 将BGR格式的图像转换为灰度图像
    gray_image = bgr2gray(image)
    # 显示转换后的灰度图像
    show_image(gray_image)

    # 将BGR格式的图像转换为灰度图像,平均值法
    gray_image = bgr2gray_avg(image)
    # 显示转换后的灰度图像
    show_image(gray_image)

    # 二值化
    binary_image = binaryzation(gray_image)
    # 显示二值化后的图像
    show_image(binary_image)

    # 保存图像
    save_image(binary_image, 'binary_lenna.png')


def main2():
    # 读取图像
    image = read_image('lenna.png')
    plt.subplot(2,2, 1)
    plt.imshow(image)
    plt.title('BGR')
    plt.axis('off')
    cv2.imwrite('BGR.png', image)

    # 将BGR格式的图像转换为RGB格式
    rgb_image = bgr2rgb(image)
    plt.subplot(2,2, 2)
    plt.imshow(rgb_image)
    plt.title('RGB')
    plt.axis('off')
    cv2.imwrite('RGB.png', rgb_image)

    # 将BGR格式的图像转换为灰度图像
    gray_image = bgr2gray(image)
    plt.subplot(2,2, 3)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gray')
    plt.axis('off')
    cv2.imwrite('Gray.png', gray_image)

    # 二值化
    binary_image = binaryzation(gray_image)
    plt.subplot(2,2, 4)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary')
    plt.axis('off')
    cv2.imwrite('Binary.png', binary_image)

    plt.show()



if __name__ == '__main__':
    # main()

    main2()





