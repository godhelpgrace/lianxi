import requests
from bs4 import BeautifulSoup

# 从网易云音乐下载歌曲
def download_song(song_id):
    url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(song_id)
    path = 'songs/{}.mp3'.format(song_id)
    return download(url, path)

# 从网易云音乐下载歌词
def download_lyric(song_id):
    url = 'http://music.163.com/api/song/lyric?os=pc&id={}&lv=-1&kv=-1&tv=-1'.format(song_id)
    r = requests.get(url)
    json_obj = r.json()
    try:
        lyric = json_obj['lrc']['lyric']
        path = 'lyrics/{}.lrc'.format(song_id)
        with open(path, 'w') as f:
            f.write(lyric)
        return True
    except:
        return False


# 从网易云音乐下载歌曲封面
def download_cover(song_id):
    url = 'http://music.163.com/api/song/detail/?id={}&ids=[{}]'.format(song_id, song_id)
    r = requests.get(url)
    json_obj = r.json()
    try:
        cover_url = json_obj['songs'][0]['album']['picUrl']
        path = 'covers/{}.jpg'.format(song_id)
        return download(cover_url, path)
    except:
        return False

# 从网易云音乐下载歌曲信息
def download_song_info(song_id):
    url = 'http://music.163.com/api/song/detail/?id={}&ids=[{}]'.format(song_id, song_id)
    r = requests.get(url)
    json_obj = r.json()
    try:
        song_name = json_obj['songs'][0]['name']
        artist_name = json_obj['songs'][0]['artists'][0]['name']
        album_name = json_obj['songs'][0]['album']['name']
        path = 'song_infos/{}.txt'.format(song_id)
        with open(path, 'w') as f:
            f.write('歌曲名：{}\n'.format(song_name))
            f.write('歌手名：{}\n'.format(artist_name))
            f.write('专辑名：{}\n'.format(album_name))
        return True
    except:
        return False

# 从网易云音乐下载歌单
def download_playlist(playlist_id):
    url = 'http://music.163.com/playlist?id={}'.format(playlist_id)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    song_ids = soup.select('#song-list-pre-cache li a')
    for song_id in song_ids:
        song_id = song_id['href'].split('=')[1]
        download_song(song_id)
        download_lyric(song_id)
        download_cover(song_id)
        download_song_info(song_id)