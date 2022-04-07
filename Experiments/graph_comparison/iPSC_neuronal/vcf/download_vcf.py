from ftplib import FTP
import pandas as pd
import numpy as np

df_eff = pd.read_csv('diff_efficiency_neur.csv')
donors = df_eff.donor_id

ftpdir = 'ftp.sra.ebi.ac.uk'

ftp = FTP(ftpdir)

ftp.login()

def check_files(filelist, preq, strlist):
    filelist = [f for f in filelist if preq in f]
    filelist = [f for f in filelist if np.any([s in f for s in strlist])]
    return len(filelist) > 0

download_list = []
for n in range(447,449):
    header=f'vol1/ERZ{n}'
    ftp.cwd(header)
    dirlist = ftp.nlst()
    for erzdir in dirlist:
        ftp.cwd(erzdir)
        filelist = ftp.nlst()
        if check_files(filelist=filelist,
                       preq='genotypes.vcf.gz',
                       strlist=donors):
            download_list.append(f'{header}/{erzdir}')
        ftp.cwd('..')
    ftp.cwd("~")

ftp.quit()

df = pd.DataFrame(download_list)
df.to_csv('vcf_ftp_list.txt',index=False,header=False)
