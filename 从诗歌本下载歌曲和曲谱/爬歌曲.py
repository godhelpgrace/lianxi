#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
from bs4 import BeautifulSoup

# 从浏览器获取图片url及歌曲url
def get_img_and_song_url(url):
    """
    从浏览器获取图片url及歌曲url
    :param url: 浏览器url
    :return: 图片url及歌曲url
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    img_url = soup.select('img')[0]['src']
    song_url = soup.select('source')[0]['src']
    return img_url, song_url

def download(url, path):
    """
    从已获取的图片url及歌曲url下载图片及歌曲
    :param url: 图片或歌曲的url
    :param path: 图片或歌曲的保存路径
    :return:
    """
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)



def get_text_from_img(img_path):
    """从图片中获取文本信息
    :param img_path: 图片路径
    :return: 文本信息
    """
    import pytesseract
    from PIL import Image
    # 从图片中识别文字
    img = Image.open(img_path)
    # 把从图片识别的文字传给变量，打印到控制台,这里的lang='chi_sim'是指定识别中文
    text = pytesseract.image_to_string(img, lang='chi_sim')
    return text

def get_song_name_from_text(text):
    """从文本中获取歌曲名
    :param text: 文本信息
    :return: 歌曲名
    """
    # 多个空格替换为一个空格
    text = ' '.join(text.split())
    # 用空格分割字符串
    text_list = text.split(' ')
    return text_list[0]

# 获取文件后缀名
def get_file_extension(file_name):
    """
    获取文件后缀名
    :param file_name: 文件名
    :return: 文件后缀名
    """
    return file_name.split('.')[-1]

def rename_file(old_path, new_path):
    """
    移动文件，如果文件存在则加 “_数字”
    :param old_path: 文件原路径
    :param new_path: 文件新路径
    :return:
    """
    if os.path.exists(new_path):
        i = 1
        while True:
            if os.path.exists(new_path.split('.')[0] + '_' + str(i) + '.' + new_path.split('.')[1]):
                i += 1
            else:
                os.rename(old_path, os.path.basename(new_path) + '_' + str(i) + '.' + new_path.split('.')[-1])
                break
    else:
        os.rename(old_path + new_path, old_path + new_path)


def move_file(old_path, new_path):
    """
    移动文件，如果文件存在则加 “_数字”
    :param old_path: 文件原路径
    :param new_path: 文件新路径
    :return:
    """
    # 判断新路径下文件是否存在
    if os.path.exists(new_path):
        i = 1
        # 如果存在，获取文件后缀名
        extension = get_file_extension(new_path)
        # 获取文件名
        file_name = os.path.basename(new_path).split('.')[0]
        # 获取父目录
        parent_path = os.path.dirname(new_path)
        new_path1 = os.path.join(parent_path, file_name + '_1.' + extension)
        # 判断文件名_1是否存在,如果存在，i加1,直到文件名_i不存在,则将文件名_i作为新文件名,并移动文件
        if os.path.exists(os.path.join(parent_path, file_name + '_1.' + extension)):
            while True:
                # 如果文件存在，i加1
                if os.path.exists(os.path.join(parent_path, file_name + '_' + str(i) + '.' + extension)):
                    i += 1
                else:
                    break
                new_path1 = os.path.join(parent_path, file_name + '_' + str(i) + '.' + extension)
        os.rename(old_path, new_path1)

def download_from_url(url):
    # 从浏览器获取图片url及歌曲url
    img_url, song_url = get_img_and_song_url(url)

    # 对url进行处理，替换 ×tamp 为 &timestamp
    img_url, song_url = img_url.replace('×tamp','&timestamp'), song_url.replace('×tamp','&timestamp')
    print(img_url, song_url)
    tmp_name = 'tmp_name'
    # 从已获取的图片url及歌曲url下载图片及歌曲
    download(img_url, tmp_name + '.jpg')
    download(song_url, tmp_name + '.mp3')
    # 从图片中获取文本信息
    text = get_text_from_img(tmp_name + '.jpg')
    # 从文本中获取歌曲名
    song_name = get_song_name_from_text(text)
    # 创建文件夹
    os.makedirs('Image', exist_ok=True)
    os.makedirs('Song', exist_ok=True)
    # 判断文件是否存在
    if os.path.exists(song_name + '.jpg'):
        os.remove(song_name + '.jpg')
    if os.path.exists(song_name + '.mp3'):
        os.remove(song_name + '.mp3')
    # 重命名文件
    os.rename(tmp_name + '.jpg', song_name + '.jpg')
    os.rename(tmp_name + '.mp3', song_name + '.mp3')

    # 移动文件，如果文件存在则加 “_数字”
    move_file(song_name + '.jpg', 'Image/' + song_name + '.jpg')
    move_file(song_name + '.mp3', 'Song/' + song_name + '.mp3')


def download_from_txt(img_url, song_url):

    tmp_name = 'tmp_name'
    # 从已获取的图片url及歌曲url下载图片及歌曲
    download(img_url, tmp_name + '.jpg')
    download(song_url, tmp_name + '.mp3')
    # 从图片中获取文本信息
    text = get_text_from_img(tmp_name + '.jpg')
    # 从文本中获取歌曲名
    song_name = get_song_name_from_text(text)
    # 创建文件夹
    os.makedirs('Image', exist_ok=True)
    os.makedirs('Song', exist_ok=True)
    # 判断文件是否存在
    if os.path.exists(song_name + '.jpg'):
        os.remove(song_name + '.jpg')
    if os.path.exists(song_name + '.mp3'):
        os.remove(song_name + '.mp3')
    # 重命名文件
    os.rename(tmp_name + '.jpg', 'Image/' + song_name + '.jpg')
    os.rename(tmp_name + '.mp3', 'Song/' + song_name + '.mp3')

    # 移动文件，如果文件存在则加 “_数字”
    move_file(song_name + '.jpg', 'Image/' + song_name + '.jpg')
    move_file(song_name + '.mp3', 'Song/' + song_name + '.mp3')


def download_from_txt2(img_url):
    tmp_name = 'tmp_name'
    # 从已获取的图片url及歌曲url下载图片及歌曲
    download(img_url, tmp_name + '.jpg')
    # 从图片中获取文本信息
    text = get_text_from_img(tmp_name + '.jpg')
    # 从文本中获取歌曲名
    song_name = get_song_name_from_text(text)
    # 创建文件夹
    os.makedirs('Image', exist_ok=True)
    # 判断文件是否存在
    if os.path.exists(song_name + '.jpg'):
        os.remove(song_name + '.jpg')
    # 重命名文件
    os.rename(tmp_name + '.jpg', 'Image/' + song_name + '.jpg')
    # 移动文件，如果文件存在则加 “_数字”
    move_file(song_name + '.jpg', 'Image/' + song_name + '.jpg')


# 读取文本文件的内容，以一行为单位，返回一个列表
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

# 读取文本文件的内容，查找指定内容，返回一个列表
def read_file(file_path, search_content):
    with open(file_path, 'r') as f:
        return [line for line in f.readlines() if search_content in line]

# 列表的元素去重
def remove_duplicate(lis):
    return list(set(lis))

if __name__ == '__main__':

    # url = 'https://12180.net/m/kte4L'
    # download_from_url(url)
    a = read_file('s.txt', '/api/download/')
    a = [i.replace('\n','') for i in a]

    for i in a:
        b = '/api/download/image?'
        if b in i:
            c = i.split(b)[0]
            d = i.split(b)[1]
            e = d.split('&')[0]
            f = d.split('&')[1]
            for j in range(1, len(a)):
                if c in a[j] and f in a[j] and i != a[j]:
                    image_url = i
                    song_url = a[j]
                    print('=====================')
                    print(image_url, song_url)
                    try:
                        download_from_txt(image_url, song_url)
                        print('ok')
                    except:
                        print('error')







