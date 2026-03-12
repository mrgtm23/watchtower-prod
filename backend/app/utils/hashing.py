import hashlib

def sha256_of_fileobj(fileobj) -> str:
    # fileobj must be seekable; read then rewind
    pos = fileobj.tell()
    fileobj.seek(0)
    h = hashlib.sha256()
    while True:
        chunk = fileobj.read(8192)
        if not chunk:
            break
        h.update(chunk)
    fileobj.seek(pos)
    return h.hexdigest()