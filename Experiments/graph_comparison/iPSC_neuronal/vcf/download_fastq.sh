mv ERR469 ERR4699947_1.fastq.gz
gunzip ERR4699947_1.fastq.gz


#!/bin/bash
START=4699947
END=4700000
# END=4700478

for (( c=$START; c<=$END; c++ ))
do
  filenum="$c"
  num1=${c:0:3}
  num2=${c: -1}
  filename="ERR${filenum}"
  dir1="ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR${num1}/00${num2}/${filename}/${filename}_1.fastq.gz"
  dir2="ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR${num1}/00${num2}/${filename}/${filename}_2.fastq.gz"
  echo $dir1
  wget --content-disposition "${dir1}"
  wget --content-disposition "${dir2}"
  gunzip "${filename}_1.fastq.gz"
  gunzip "${filename}_2.fastq.gz"
done

# For reference genome
wget -r ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/
