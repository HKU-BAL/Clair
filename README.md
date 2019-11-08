# Clair - Yet another deep neural network based variant caller
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) \
Contact: Ruibang Luo \
Email: rbluo@cs.hku.hk

## Introduction
__Clair__ is the successor of [Clairvoyante](https://github.com/aquaskyline/clairvoyante). \
\
__[TODO] introduction__

---

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Submodule Descriptions](#submodule-descriptions)
- [Download Pretrained Models](#pretrained-models)
- [Advanced Guides](#advanced-guides)
- [[TODO] Train](#todo-train)

---

## Installation

### Option 1. conda for virtual environment
#### If anaconda3 not installed, checkout https://docs.anaconda.com/anaconda/install/ for the installation guide
```bash
# create and activate the environment named clair
conda create —name clair python=3.7
conda activate clair

# install pypy and packages on clair environemnt
conda install -c conda-forge pypy3.6
pypy3 -m ensurepip
pypy3 -m pip install blosc
pypy3 -m pip install intervaltree

# install python packages on clair environment
pip install numpy blosc intervaltree tensorflow==1.13.2 pysam
conda install pigz

# clone Clair
git clone --depth=1 https://github.com/HKU-BAL/Clair.git
cd Clair

# download pretrained model (for Illumina)
mkdir illumina && cd illumina
wget http://www.bio8.cs.hku.hk/clair_models/illumina/trained_models.tar
tar -xf trained_models.tar
cd ../

# download pretrained model (for PacBio CCS)
mkdir pacbio && cd pacbio
wget http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/trained_models.tar
tar -xf trained_models.tar
cd ../

# download pretrained model (for ONT)
mkdir ont && cd ont
wget http://www.bio8.cs.hku.hk/clair_models/ont/trained_models.tar
tar -xf trained_models.tar
cd ../
```

### [TODO] Option 2. Bioconda

### [TODO] Option 3. Docker

### After Installation

To check the version of Tensorflow you have installed:
```bash
python -c 'import tensorflow as tf; print(tf.__version__)'
```

To do variant calling using trained models, CPU will suffice. Clair uses all available CPU cores by default in `call_var`, 4 threads by default in `callVarBam`. The number of threads to be used can be controlled using the parameter `--threads`. To train a new model, a high-end GPU and the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:

```bash
pip install tensorflow-gpu==1.13.2
```

The installation of the `blosc` library might fail if your CPU doesn't support the AVX2 instruction set. Alternatively, you can compile and install from the latest source code available in [GitHub](https://github.com/Blosc/python-blosc) with the "DISABLE_BLOSC_AVX2" environment variable set.

---

## Usage

### General usage
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"

# to run a submodule using python
python $CLAIR [submodule] [options]

# to run a PyPy-able submodule using pypy
pypy3 $CLAIR [submodule] [options]
```

<h2>

### Setup variables for variant calling commands afterwards
```bash
CLAIR="[PATH_TO_CLAIR]/clair.py"                         # e.g. ./clair.py
MODEL="[MODEL_PATH]"                                     # e.g. [PATH_TO_CLAIR]/ont/model
BAM_FILE_PATH="[YOUR_BAM_FILE]"                          # e.g. chr21.bam
REFERENCE_FASTA_FILE_PATH="[YOUR_REFERENCE_FASTA_FILE]"  # e.g. chr21.fa
BED_FILE_PATH="[YOUR_BED_FILE]"                          # e.g. chr21.bed
PYPY="[PYPY_BIN_PATH]"                                   # e.g. pypy
```
#### Note
* For `PYPY` variable, if installed using installation option 1, use `PYPY="pypy3"`
* For `MODEL` variable, no need to type anything after `model`<br>(if you are using pretrained model with 3 files named _model.data-00000-of-00001_, _model.index_, _model.meta_)

<h2>

### Call variants at known variant sites (using `callVarBam`)
```bash
# variables
VARIANT_CALLING_OUTPUT_PATH="[YOUR_OUTPUT_PATH]"         # e.g. chr21.vcf (make sure the directory exists)
CONTIG_NAME="[CONTIG_NAME_FOR_VARIANT_CALLING]"          # e.g. chr21

python $CLAIR callVarBam \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bed_fn "$BED_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--call_fn "$VARIANT_CALLING_OUTPUT_PATH" \
--pypy "$PYPY" \
--ctgName "$CONTIG_NAME"

less "$VARIANT_CALLING_OUTPUT_PATH"
```

#### Note
* For variant calling using Illumina or PacBio Data (or ONT Data with fewer # of positions in BED file), you may consider to add `--pysam_for_all_indel_bases` for more accurate results. (This option need spend more time on ONT data, thus not advice to use this option on ONT whole genome variant calling)
* If you are interested in allele frequency filtering, check [About Setting the Alternative Allele Frequency Cutoff](#about-setting-the-alternative-allele-frequency-cutoff)

<h2>

### Call variants from BAM in parallel (using `callVarBamParallel`)
```bash
# variables
SAMPLE_NAME="call"
OUTPUT_PREFIX="tmp"

# create command.sh for run jobs in parallel
python $CLAIR callVarBamParallel \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bed_fn "$BED_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--pypy "$PYPY" \
--sampleName="$SAMPLE_NAME" > command.sh

export CUDA_VISIBLE_DEVICES=""
cat command.sh | parallel -j4

# concat vcf(s) and sort the variants called
vcfcat tmp*.vcf | vcfstreamsort | bgziptabix snp_and_indel.vcf.gz
```

#### Note
* `callVarBamParallel` submodule is to generate `callVarBam` commands that can be run in [parallel](https://www.gnu.org/software/parallel/)
* `parallel -j4` will run 4 commands in parallel. Advice to choose the number equal to # of cores in the CPU (or highest # of threads can be handled for the CPU in case the CPU support hyper-threading).
* If [parallel](https://www.gnu.org/software/parallel/) not installed, try ```awk '{print "\""$0"\""}' commands.sh | xargs -P4 -L1 sh -c```
* If no BED file was provided, Clair will call variants on the whole genome.
* `vcfcat`, `vcfstreamsort` and `bgziptabix` are a part of [vcflib](https://github.com/vcflib/vcflib).
* `CUDA_VISIBLE_DEVICES=""` makes GPUs invisible to Clair so it will use CPU only. Please notice that unless you want to run `commands.sh` in serial, you cannot use GPU because one running copy of Clairvoyante will occupy all available memory of a GPU. While the bottleneck of `callVarBam` is at the CPU only `CreateTensor` script, the effect of GPU accelerate is insignificant (roughly about 15% faster). But if you have multiple GPU cards in your system, and you want to utilize them in variant calling, you may want split the `commands.sh` in to parts, and run the parts by firstly `export CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the seqeunce of the GPU to be used.
* If you are going to call on non-human BAM file (e.g. bacteria), add `--includingAllContigs` option to call on contigs besides chromosome 1-22/X/Y/M/MT
* You may also check the Notes in above sections for common considerations.

---

## Submodule Descriptions

There are two separated folders storing different submodules for different purposes. \
__`clair/`__ is for variant calling and model training submodules, and __`dataPrepScripts`__ is for dataset building submodules.

*For the submodule lists below, you can also run the program with `-h or --help` option for arguments details.*

`clair/` | Note: submodules under this folder is `pypy` incompatiable, please run in `python`
---: | ---
`call_var` | Call variants using candidate variant tensors.
`callVarBam` | Call variants directly from a BAM file.
`callVarBamParallel` | Generate `callVarBam` commands that can be run in [parallel](https://www.gnu.org/software/parallel/). A BED file is required to specify the regions for variant calling. `--refChunkSize` set the genome chuck size per job.
`evaluate` | Evaluate a model.
`plot_tensor` | Create high resolution PNG figures to visualize input tensor.
`train` |  Training a model using adaptive learning rate decay. By default, the learning rate will decay for three times. Input a binary tensors file created by `Tensor2Bin` is highly recommended.
`train_clr` | Training a model using Cyclical Learning Rate (CLR).


`dataPrepScripts/` | Note: submodules under this folder is `pypy` compatiable unless specified.
---: | ---
`ExtractVariantCandidates`| Extract the position of variant candidates. Input: BAM; Reference FASTA.<br>_Important option(s):<br>`--threshold` "Minimum alternative allelic fraction to report a candidate"<br>`--minCoverage` "Minimum coverage to report a candidate"_
`GetTruth`| Extract the variants from a truth VCF. Input: VCF.
`CreateTensor`| Create tensors for candidates or truth variants. Input: A candidate list; BAM; Reference FASTA.<br>_Important option(s):<br>`--stop_consider_left_edge`. Negation to consider left edge: Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor.<br>Consider left edge if you are:<br>1) using reads shorter than 100bp<br>2) using a tensor with flanking length longer than 16bp<br>3) you are using amplicon sequencing or other sequencing technologies, in which reads starting positions are random is not a basic assumption_
`PairWithNonVariants`| Pair truth variant tensors with non-variant tensors. Input: Truth variants tensors; Candidate variant tensors.<br>_Important option(s):<br>`--amp x` "1-time truth variants + x-time non-variants"._
`Tensor2Bin` | Create a compressed binary tensors file to facilitate and speed up future usage. Input: Mixed tensors by `PairWithNonVariants`; Truth variants by `GetTruth` and a BED file marks the high confidence regions in the reference genome.<br>(`pypy` incompatible)
`CombineBins` | Combine small bins from `Tensor2Bin` into a single large bin.<br>(`pypy` incompatible)

---

## Pretrained Models

Please download models from [here](http://www.bio8.cs.hku.hk/clair_models/) or click the below links to download.

Folder | Tech | Sample used | Aligner | Ref | Download |
--- | :---: | :---: | :---: | :---: | :---: |
illumina | Illumina | HG001,2,3,4,5 | Novoalign | hg38 | [Download](http://www.bio8.cs.hku.hk/clair_models/illumina/trained_models.tar)
pacbio/ccs | PacBio CCS 15k | HG001,5 | Minimap2 | hg19 | [Download](http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/trained_models.tar)
ont | ONT | HG001,2 | Minimap2 | hg38 | [Download](http://www.bio8.cs.hku.hk/clair_models/ont/trained_models.tar)

---

## Advanced Guides


### About Setting the Alternative Allele Frequency Cutoff
Different from model training, in which all genome positions are candidates but randomly subsampled for training, variant calling using a trained model will require the user to define a minimal alternative allele frequency cutoff for a genome position to be considered as a candidate for variant calling. For all sequencing technologies, the lower the cutoff, the lower the speed. Setting a cutoff too low will increase the false positive rate significantly, while too high will increase the false negative rate significantly. \
The option `--threshold` controls the cutoff in these submodules `callVarBam`, `callVarBamParallel` and `ExtractVariantCandidates`. The suggested cutoff is listed below for different sequencing technologies. A higher cutoff will increase the accuracy of datasets with poor sequencing quality, while a lower cutoff will increase the sensitivity in applications like clinical research. Setting a lower cutoff and further filter the variants by their quality is also a good practice.

Sequencing Technology | Alt. AF Cutoff |
:---: |:---:|
Illumina | 0.1 |
PacBio | 0.2 |
ONT | 0.2 |


### Speed up with PyPy
Without a change to the code, using PyPy python interpreter on some tensorflow independent modules such as `ExtractVariantCandidates` and `CreateTensor` gives a 5-10 times speed up. Pypy python interpreter can be installed by apt-get, yum, Homebrew, MacPorts, etc. If you have no root access to your system, the official website of Pypy provides a portable binary distribution for Linux. Beside following the conda installation method in [Installation](#installation), the following is a rundown extracted from Pypy's website (PyPy3.6 v7.2.0 in this case) on how to install the binaries.

```bash
wget https://github.com/squeaky-pl/portable-pypy/releases/download/pypy3.6-7.2.0/pypy3.6-7.2.0-linux_x86_64-portable.tar.bz2
tar -jxf pypy3.6-7.2.0-linux_x86_64-portable.tar.bz2
cd pypy3.6-7.2.0-linux_x86_64-portable/bin
./pypy -m ensurepip
./pip install -U pip wheel intervaltree
# Use pypy as an inplace substitution of python to run PyPy-able scripts
```

Alternatively, if you can use apt-get or yum in your system, please install both `pypy` and `pypy-dev` packages. And then install the pip for pypy.

```bash
sudo apt-get install pypy pypy-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo pypy get-pip.py
sudo pypy -m pip install intervaltree
```

To guarantee a good user experience (good speed), pypy must be installed to run `callVarBam` (call variants from BAM), and `callVarBamParallel` that generate parallelizable commands to run `callVarBam`.
Tensorflow is optimized using Cython thus not compatible with `pypy`. For the list of scripts compatible to `pypy`, please refer to the [Submodule Descriptions](#submodule-descriptions).
*Pypy is an awesome Python JIT intepreter, you can donate to [the project](https://pypy.org).*


---

## [TODO] Train
