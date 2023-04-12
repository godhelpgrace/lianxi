import os

# 文件更名
def change_name(oldname, newname):
    path1 = 'Image/'
    file_list1 = os.listdir(path1)
    for file in file_list1:
        if file == oldname:
            print('更名前的文件名为：' + file)
            os.rename(path1 + '/' + file, path1 + '/' + newname)
            print('更名成功！')

    path2 = 'Song/'
    file_list2 = os.listdir(path2)
    for file in file_list2:
        if file == oldname:
            print('更名前的文件名为：' + file)
            os.rename(path2 + '/' + file, path2 + '/' + newname)
            print('更名成功！')



if __name__ == '__main__':
    old = '走'
    new = '走走走'
    change_name(oldname=old+'.jpg', newname=new+'.jpg')
    change_name(oldname=old+ '.mp3', newname=new + '.mp3')