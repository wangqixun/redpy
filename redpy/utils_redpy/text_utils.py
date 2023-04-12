

def is_zh(ch):
    return u'\u4e00' <= ch <= u'\u9fff'

def has_zh(string):
    return any(map(is_zh, string))
