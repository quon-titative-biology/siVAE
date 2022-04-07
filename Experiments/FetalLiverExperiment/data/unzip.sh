#!/bin/bash

mkdir ArrayExpress

unzip zipfiles/E-MTAB-7407.processed.1.zip -d ArrayExpress
unzip zipfiles/E-MTAB-7407.processed.2.zip -d ArrayExpress
unzip zipfiles/E-MTAB-7407.processed.3.zip -d ArrayExpress
unzip zipfiles/E-MTAB-7407.processed.4.zip -d ArrayExpress

cd ArrayExpress
for a in $(ls -1 *.tar.gz); do tar -zxvf $a; done
