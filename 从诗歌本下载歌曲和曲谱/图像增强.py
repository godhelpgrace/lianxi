

# 增加图片清晰度,使用CV2
def enhance_image(img_path):
    import cv2
    import numpy as np
    # 读取图片
    img = cv2.imread(img_path)
    # 图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)



    # 图像增强，使用直方图均衡化
    dst = cv2.equalizeHist(binary)

    # # 图像增强，使用拉普拉斯算子
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # dst = cv2.filter2D(binary, -1, kernel=kernel)
    #
    # # # 图像增强，使用伽马变换
    # # gamma = 1.5
    # # dst = np.power(binary / float(np.max(binary)), gamma) * 255.0

    # 提高图片分辨率
    dst = cv2.resize(dst, (dst.shape[1] * 4, dst.shape[0] * 4), interpolation=cv2.INTER_CUBIC)

    # 图像增强，使用高斯滤波
    dst = cv2.GaussianBlur(dst, (5, 5), 0)



    # 保存图片
    cv2.imwrite('enhance_' + img_path, dst)


    # # 图片腐蚀
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(binary, kernel, iterations=1)
    # # 图片膨胀
    # dilation = cv2.dilate(erosion, kernel, iterations=1)
    # # 保存图片
    # cv2.imwrite('enhance_' + img_path, dilation)

if __name__ == '__main__':
    img_path = 'ys.jpg'
    enhance_image(img_path)




