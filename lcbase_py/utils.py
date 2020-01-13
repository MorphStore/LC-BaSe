"""
Some little utilities.
"""

def getFromDicts(dictExcept, dictMain, key):
    if dictExcept is not None and key in dictExcept:
        return dictExcept[key]
    else:
        return dictMain.get(key, key)