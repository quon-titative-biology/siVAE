grep '^MT' HPSI0214i-kucg_2.wgs.gatk.haplotype_caller.20170425.genotypes.vcf > MT.vcf

find . -name "*.vcf"

mkdir MTvcf

for f in $(find . -name "*genotypes.vcf")
do
    filename="$(basename $f)"
    task="$(cut -d"." -f1 <<< $filename)"
    # basedir="$(dirname $f)"
    MTfilename="$task.vcf"
    echo $MTfilename
    grep "^MT" $f > "MTvcf/$MTfilename"
done
