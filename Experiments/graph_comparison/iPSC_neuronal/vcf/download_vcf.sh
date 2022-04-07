mkdir vcf
mv vcf

# Generate vcf_ftp_list.txt
python download_vcf.sh

while IFS='' read -r LINE || [ -n "${LINE}" ]; do
  filename="ftp://ftp.sra.ebi.ac.uk/${LINE}"
  echo $filename
  wget -r $filename
done < vcf_ftp_list.txt


while IFS='' read -r LINE || [ -n "${LINE}" ]; do
  filename="ftp.sra.ebi.ac.uk/${LINE}"
  cd $filename
  # gunzip *.gz
  cd ../../../..
done < vcf_ftp_list.txt

find . -name \*.gz

gunzip -r .
