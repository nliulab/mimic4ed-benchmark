import gzip
import shutil
from os import listdir
from os.path import isfile, join
import IPython
def unzip_files(dir):
    gzfiles = [join(dir, f) for f in listdir(dir) if '.gz' in f and isfile(join(dir, f))]
    for f in gzfiles:
        with gzip.open(f, 'rb') as f_in:
            with open(f[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# display large dataframes in an html iframe
def ldf_display(df, lines=500):
    txt = ("<iframe " +
           "srcdoc='" + df.head(lines).to_html() + "' " +
           "width=1000 height=500>" +
           "</iframe>")

    return IPython.display.HTML(txt)