# Clair - Yet another deep neural network based variant caller
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/clair/README.html)  
Contact: Ruibang Luo  
Email: rbluo@cs.hku.hk  

## Introduction
Single-molecule sequencing technologies have emerged in recent years and revolutionized structural variant calling, complex genome assembly, and epigenetic mark detection. However, the lack of a highly accurate small variant caller has limited the new technologies from being more widely used. In this study, we present Clair, the successor to Clairvoyante, a program for fast and accurate germline small variant calling, using single molecule sequencing data. For ONT data, Clair achieves the best precision, recall and speed as compared to several competing programs, including Clairvoyante, Longshot and Medaka. Through studying the missed variants and benchmarking intentionally overfitted models, we found that Clair may be approaching the limit of possible accuracy for germline small variant calling using pileup data and deep neural networks.

This is the formal release of Clair (Clair v2, Dec 2019). You can find the experimental Clair v1 (Jan 2019) at [https://github.com/aquaskyline/Clair](https://github.com/aquaskyline/Clair). The preprint of Clair v2 is available in [bioAxiv](https://biorxiv.org/cgi/content/short/865782v1).

---

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Submodule Descriptions](#submodule-descriptions)
- [Download Pretrained Models](#pretrained-models)
- [Advanced Guides](#advanced-guides)
- [Model Training](docs/TRAIN.md)

---

## Installation

### Option 1. conda for virtual environment
#### If anaconda3 not installed, checkout https://docs.anaconda.com/anaconda/install/ for the installation guide
```bash
# create and activate the environment named clair
conda create -n clair python=3.7
conda activate clair

# install pypy and packages on clair environemnt
conda install -c conda-forge pypy3.6
pypy3 -m ensurepip
pypy3 -m pip install blosc intervaltree

# install python packages on clair environment
pip install numpy blosc intervaltree tensorflow==1.13.2 pysam matplotlib
conda install -c anaconda pigz
conda install -c conda-forge parallel zstd
conda install -c bioconda samtools vcflib

# install vcftools
sudo apt-get install vcftools

# clone Clair
git clone --depth=1 https://github.com/HKU-BAL/Clair.git
cd Clair

# download pretrained model (for ONT)
mkdir ont && cd ont
wget http://www.bio8.cs.hku.hk/clair_models/ont/12.tar
tar -xf 12.tar
cd ../

# download pretrained model (for PacBio CCS)
mkdir pacbio && cd pacbio
wget http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/15.tar
tar -xf 15.tar
cd ../

# download pretrained model (for Illumina)
mkdir illumina && cd illumina
wget http://www.bio8.cs.hku.hk/clair_models/illumina/12345.tar
tar -xf 12345.tar
cd ../
```

### Option 2. Bioconda

```bash
# make sure channels are added in conda
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge

# create conda environment named "clair-env"
conda create -n clair-env -c bioconda clair
conda activate clair-env

# install vcftools
sudo apt-get install vcftools

# store clair.py PATH into $CLAIR variable
CLAIR=`which clair.py`

# run clair like this afterwards
python $CLAIR --help
```

The conda environment has the Pypy intepreter installed, but two Pypy libraries `intervaltree` and `blosc` are still missing. The reason why the two packages are not installed by default is because they are not yet available in any conda repositories. To install the two libraries for Pypy, after activating the conda environment, please run the follow commands:

```bash
pypy3 -m ensurepip
pypy3 -m pip install --no-cache-dir intervaltree blosc
```

Download the models to a folder and continue the process
(Please refer to `# download pretrained model` in  [Installation Option 1](#option-1-conda-for-virtual-environment))


### Option 3. Docker
```bash
# clone Clair
git clone --depth=1 https://github.com/HKU-BAL/Clair.git
cd Clair

# download pretrained model (for ONT)
mkdir ont && cd ont
wget http://www.bio8.cs.hku.hk/clair_models/ont/12.tar
tar -xf 12.tar
cd ../

# download pretrained model (for PacBio CCS)
mkdir pacbio && cd pacbio
wget http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/15.tar
tar -xf 15.tar
cd ../

# download pretrained model (for Illumina)
mkdir illumina && cd illumina
wget http://www.bio8.cs.hku.hk/clair_models/illumina/12345.tar
tar -xf 12345.tar
cd ../

# build a docker image named clair_docker_image
docker build -f ./Dockerfile -t clair_docker_image .

# run docker image
docker run -it clair_docker_image

# store clair.py PATH into $CLAIR variable
CLAIR=`which clair.py`

# run clair like this afterwards
python $CLAIR --help
```

### After Installation

To check the version of Tensorflow you have installed:

```bash
python -c 'import tensorflow as tf; print(tf.__version__)'
```

To do variant calling using trained models, CPU will suffice. Clair uses 4 threads by default in `callVarBam`. The number of threads to be used can be controlled using the parameter `--threads`. To train a new model, a high-end GPU and the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:

```bash
pip install tensorflow-gpu==1.13.2
```

The installation of the `blosc` library might fail if your CPU doesn't support the AVX2 instruction set. Alternatively, you can compile and install from the latest source code available in [GitHub](https://github.com/Blosc/python-blosc) with the `DISABLE_BLOSC_AVX2` environment variable set.

---

## Usage

### General usage
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"

# to run a submodule using python
python $CLAIR [submodule] [options]

# to run a Pypy-able submodule using pypy (if `pypy3` is the executable command for Pypy)
pypy3 $CLAIR [submodule] [options]
```

### Setup variables for variant calling commands afterwards

```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                         # e.g. clair.py
MODEL="[MODEL_PATH]"                                     # e.g. [PATH_TO_CLAIR_MODEL]/ont/model
BAM_FILE_PATH="[YOUR_BAM_FILE]"                          # e.g. chr21.bam
REFERENCE_FASTA_FILE_PATH="[YOUR_REFERENCE_FASTA_FILE]"  # e.g. chr21.fa
KNOWN_VARIANTS_VCF="[YOUR_VCF_FILE]"                     # e.g. chr21.vcf
```

#### Note
* Each model has three files `model.data-00000-of-00001`, `model.index`, `model.meta`. For the `MODEL` variable, please use the prefix `model`

### Call variants at known variant sites or in a chromosome  (using `callVarBam`)

#### Call variants in a chromosome
```bash
# variables
VARIANT_CALLING_OUTPUT_PATH="[YOUR_OUTPUT_PATH]"         # e.g. calls/chr21.vcf (please make sure the directory exists)
CONTIG_NAME="[CONTIG_NAME_FOR_VARIANT_CALLING]"          # e.g. chr21
SAMPLE_NAME="[SAMPLE_NAME]"                              # e.g. HG001

python $CLAIR callVarBam \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--ctgName "$CONTIG_NAME" \
--sampleName "$SAMPLE_NAME" \
--call_fn "$VARIANT_CALLING_OUTPUT_PATH"

cd "$VARIANT_CALLING_OUTPUT_PATH"
```

#### Call variants at known variant sites in a chromosome
```bash
# variables
VARIANT_CALLING_OUTPUT_PATH="[YOUR_OUTPUT_PATH]"         # e.g. calls/chr21.vcf (please make sure the directory exists)
CONTIG_NAME="[CONTIG_NAME_FOR_VARIANT_CALLING]"          # e.g. chr21
SAMPLE_NAME="[SAMPLE_NAME]"                              # e.g. HG001
KNOWN_VARIANTS_VCF="[YOUR_VCF_PATH]"                     # e.g. chr21_candidates.vcf

python $CLAIR callVarBam \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--ctgName "$CONTIG_NAME" \
--sampleName "$SAMPLE_NAME" \
--vcf_fn "$KNOWN_VARIANTS_VCF" \
--call_fn "$VARIANT_CALLING_OUTPUT_PATH" \

cd "$VARIANT_CALLING_OUTPUT_PATH"
```

#### Notes
* Use `callVarBam` to either call variant in a single chromosome. For whole genome variant calling, please use `callVarBamParallel` to generate multiple commands that invokes `callVarBam` on smaller chromosome chucks.
* You may consider using the `--pysam_for_all_indel_bases` option for more accurate results. On Illumina data and PacBio CCS data, the option requires 20% to 50% much running time. On ONT data, Clair can run two times slower, while the improvement in accuracy is not significant.
* About seting an appropriate allele frequency cutoff, please refer to [About Setting the Alternative Allele Frequency Cutoff](#about-setting-the-alternative-allele-frequency-cutoff)

### Call whole-genome variants in parallel (using `callVarBamParallel`)
```bash
# variables
SAMPLE_NAME="NA12878"
OUTPUT_PREFIX="call/var"                        # please make sure the call/ directory exists

# create command.sh for run jobs in parallel
python $CLAIR callVarBamParallel \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--threshold 0.2 \
--sampleName "$SAMPLE_NAME" \
--output_prefix "$OUTPUT_PREFIX" > command.sh

# disable GPU if you have one installed
export CUDA_VISIBLE_DEVICES=""

# run Clair with 4 concurrencies
cat command.sh | parallel -j4

# Find incomplete VCF files and rerun them
for i in OUTPUT_PREFIX.*.vcf; do if ! [ -z "$(tail -c 1 "$i")" ]; then echo "$i"; fi ; done | grep -f - command.sh | sh

# concatenate vcf files and sort the variants called
vcfcat ${OUTPUT_PREFIX}.*.vcf | vcf-sort -c | bgziptabix snp_and_indel.vcf.gz
```

#### Note
* `callVarBamParallel` submodule generates `callVarBam` commands that can be run in parallel
* `parallel -j4` will run four concurrencies in parallel using GNU parallel. We suggest using half the number of available CPU cores (not threads).
* If [GNU parallel](https://www.gnu.org/software/parallel/) is not installed, please try ```awk '{print "\""$0"\""}' commands.sh | xargs -P4 -L1 sh -c```
* Incomplete VCF files happens when 'out of memory' or other errors occur. The command in the example finds for a newline at the end of the VCF files, and regenerate the files without a newline at the end.
* callVarBamParallel will generate commonds for chr{1..22},X,Y, to call variants on all chromosomes, please use option `--includingAllContigs`.
* If you are working on non-human BAM file (e.g. bacteria), please use `--includingAllContigs` option to include all contigs
* `CUDA_VISIBLE_DEVICES=""` makes GPUs invisible to Clair so it will use CPU for variant calling. Please notice that unless you want to run `commands.sh` in serial, you cannot use GPU because one running copy of Clair will occupy all available memory of a GPU. While the bottleneck of `callVarBam` is at the `CreateTensor` script, which runs on CPU, the effect of GPU accelerate is insignificant (roughly about 15% faster). But if you have multiple GPU cards in your system, and you want to utilize them in variant calling, you may want split the `commands.sh` in to parts, and run the parts by firstly `export CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the ID of the GPU to be used.
* `vcfcat` and `bgziptabix` commands are from [vcflib](https://github.com/vcflib/vcflib), and are installed by default using option 2 (conda) or option 3 (docker).
* `vcf-sort` command is from [vcftools](https://github.com/vcftools/vcftools)
* Please also check the notes in the above sections for other considerations.

---

## Submodule Descriptions

Submodules in __`clair/`__ are for variant calling and model training. Submodules in __`dataPrepScripts`__ are for data preparation.

*For the submodules listed below, you use the `-h` or `--help` option for available options.*

`clair/` | Note: submodules under this folder is Pypy incompatiable, please run using Python
---: | ---
`call_var` | Call variants using candidate variant tensors.
`callVarBam` | Call variants directly from a BAM file.
`callVarBamParallel` | Generate `callVarBam` commands that can be run in parallel. A BED file is required to specify the regions for variant calling. `--refChunkSize` set the genome chuck size per job.
`evaluate` | Evaluate a model.
`plot_tensor` | Create high resolution PNG figures to visualize input tensor.
`train` |  Training a model using adaptive learning rate decay. By default, the learning rate will decay for three times. Input a binary tensors file created by `Tensor2Bin` is highly recommended.
`train_clr` | Training a model using Cyclical Learning Rate (CLR).


`dataPrepScripts/` | Note: submodules under this folder is Pypy compatiable unless specified.
---: | ---
`ExtractVariantCandidates`| Extract the position of variant candidates.<br>Input: BAM; Reference FASTA.<br>_Important option(s):<br>`--threshold` "Minimum alternative allele frequency to report a candidate"<br>`--minCoverage` "Minimum coverage to report a candidate"_
`GetTruth`| Extract the variants from a truth VCF. Input: VCF.
`CreateTensor`| Create tensors for candidates or truth variants.<br>Input: A candidate list; BAM; Reference FASTA.
`PairWithNonVariants`| Pair truth variant tensors with non-variant tensors.<br>Input: Truth variants tensors; Candidate variant tensors.<br>_Important option(s):<br>`--amp x` "1-time truth variants + x-time non-variants"._
`Tensor2Bin` | Create a compressed binary tensors file to facilitate and speed up future usage.<br>Input: Mixed tensors by `PairWithNonVariants`; Truth variants by `GetTruth` and a BED file marks the high confidence regions in the reference genome.<br>(Pypy incompatible)
`CombineBins` | Merge smaller bins from `Tensor2Bin` into a complete larger bin.<br>(Pypy incompatible)

---

## Pretrained Models

Please download models from [here](http://www.bio8.cs.hku.hk/clair_models/) or click on the links below.

Folder | Tech | Sample used | Aligner | Download |
--- | :---: | :---: | :---: | :---: |
illumina | Illumina | HG001,2,3,4,5 | Novoalign | [Download](http://www.bio8.cs.hku.hk/clair_models/illumina/12345.tar)
pacbio/ccs | PacBio CCS | HG001,5 | Minimap2 | [Download](http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/15.tar)
ont | ONT R9.4.1 | HG001,2 | Minimap2 | [Download](http://www.bio8.cs.hku.hk/clair_models/ont/12.tar)

---

## Advanced Guides


### About Setting the Alternative Allele Frequency Cutoff
Different from model training, in which all genome positions are candidates but randomly subsampled for training, variant calling using a trained model will require the user to define a minimal alternative allele frequency cutoff for a genome position to be considered as a candidate for variant calling. For all sequencing technologies, the lower the cutoff, the lower the speed. Setting a cutoff too low will increase the false positive rate significantly, while too high will increase the false negative rate significantly. \
The option `--threshold` controls the cutoff in these submodules `callVarBam`, `callVarBamParallel` and `ExtractVariantCandidates`. The suggested cutoff is listed below for different sequencing technologies. A higher cutoff will increase the accuracy of datasets with poor sequencing quality, while a lower cutoff will increase the sensitivity in applications like clinical research. Setting a lower cutoff and further filter the variants by their quality is also a good practice.

Sequencing Technology | Alt. AF Cutoff |
:---: |:---:|
Illumina | 0.1 |
PacBio CCS | 0.2 |
ONT | 0.2 |

### Variant quality cutoff selection

#### ONT data
The variant quality distribution of Clair on ONT data is usually bimodal. The best quality cutoff is usually the valley between two peaks plus 50. The image below shows the quality distribution of the variants in HG002 called using ~50-fold coverage ONT data. The best quality cutoff is 748.

![](docs/QualDist-ONT.png)

#### PacBio CCS data
The image below shows the quality distribution of the variants in HG005 called using ~30-fold coverage PacBio CCS data. The best quality cutoff is 143.

![](docs/QualDist-PBCCS.png)

#### Illumina data
The image below shows the quality distribution of the variants in HG002 called using ~60-fold coverage Illumina data. The best quality cutoff is 113.

![](docs/QualDist-ILMN.png)

### Clair uses PyPy for speedup
Without a change to the code, using PyPy python interpreter on some tensorflow independent modules such as `ExtractVariantCandidates` and `CreateTensor` gives a 5-10 times speed up. Pypy python interpreter can be installed by apt-get, yum, Homebrew, MacPorts, etc. If you have no root access to your system, the official website of Pypy provides a portable binary distribution for Linux. Beside following the conda installation method in [Installation](#installation), the following is a rundown extracted from Pypy's website (PyPy3.6 v7.2.0 in this case) on how to install the binaries.

```bash
wget https://github.com/squeaky-pl/portable-pypy/releases/download/pypy3.6-7.2.0/pypy3.6-7.2.0-linux_x86_64-portable.tar.bz2
tar -jxf pypy3.6-7.2.0-linux_x86_64-portable.tar.bz2
cd pypy3.6-7.2.0-linux_x86_64-portable/bin
./pypy3 -m pip install -U pip wheel intervaltree
# Use pypy3 as an inplace substitution of python to run pypy-able scripts
```

To guarantee a good user experience (good speed), pypy must be installed to run `callVarBam` (call variants from BAM), and `callVarBamParallel` that generate parallelizable commands to run `callVarBam`.
Tensorflow is optimized using Cython thus not compatible with `pypy3`. For the list of scripts compatible to `pypy3`, please refer to the [Submodule Descriptions](#submodule-descriptions).

*Pypy is an awesome Python JIT intepreter, you can donate to [the project](https://pypy.org).*
