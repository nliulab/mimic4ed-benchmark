import gzip
import shutil
from os import listdir
from os.path import isfile, join
def unzip_files(dir):
    gzfiles = [join(dir, f) for f in listdir(dir) if '.gz' in f and isfile(join(dir, f))]
    for f in gzfiles:
        with gzip.open(f, 'rb') as f_in:
            with open(f[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

