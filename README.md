# Clair - Yet another deep neural network based variant caller
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) <br>
Contact: Ruibang Luo <br>
Email: rbluo@cs.hku.hk

## Introduction
Single Molecule Sequencing technologies have emerged in recent years and revolutionized structural variant calling and complex genome assembly. However, the lack of a performant small variant caller has limited the new technologies from being more widely used. In this study, we present Clair, the successor of [Clairvoyante](https://github.com/aquaskyline/clairvoyante), for fast and accurate germline small variant calling using Single Molecule Sequencing data. On ONT data, Clair has achieved the best precision, recall, and speed compare to not only Clairvoyante, but also Longshot and Medaka. Through studying the failed variants and benchmarking on intentionally overfitted models, we found Clair is approaching the limit of using pileup data and deep neural network for germline small variant calling. Clair requires only CPU for variant calling.

---

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Submodule Descriptions](#submodule-descriptions)
- [Download Pretrained Models](#pretrained-models)
- [Advanced Guides](#advanced-guides)
- [[TODO] Model Training](#todo-model-training)

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
pip install numpy blosc intervaltree tensorflow==1.13.2 pysam
conda install pigz
conda install -c bioconda samtools

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

# use `clair.py` instead of `python clair.py`, the same afterwards
clair.py --help
```

The conda environment has the Pypy intepreter installed, but two Pypy libraries `intervaltree` and `blosc` are still missing. The reason why the two packages are not installed by default is because they are not yet available in any conda repositories. To install the two libraries for Pypy, after activating the conda environment, please run the follow commands:

```bash
wget https://bootstrap.pypa.io/get-pip.py
pypy3 -m ensurepip
pypy3 -m pip install --no-cache-dir intervaltree blosc
```

Download the models to a folder and continue the process
(Please refer to `# download pretrained model` in  [Installation Option 1](#option-1-conda-for-virtual-environment))


### [TODO] Option 3. Docker

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
CLAIR="[PATH_TO_CLAIR]/clair.py"                         # e.g. ./clair.py
MODEL="[MODEL_PATH]"                                     # e.g. [PATH_TO_CLAIR]/ont/model
BAM_FILE_PATH="[YOUR_BAM_FILE]"                          # e.g. chr21.bam
REFERENCE_FASTA_FILE_PATH="[YOUR_REFERENCE_FASTA_FILE]"  # e.g. chr21.fa
BED_FILE_PATH="[YOUR_BED_FILE]"                          # e.g. chr21.bed
PYPY="[PYPY_BIN_PATH]"                                   # e.g. pypy3
```

#### Note
* For the `PYPY` variable, if installed using installation option 1 or 2, use `PYPY="pypy3"`
* Each model has three files `model.data-00000-of-00001`, `model.index`, `model.meta`. For the `MODEL` variable, please use the prefix `model`

### Call variants at known variant sites (using `callVarBam`)

```bash
# variables
VARIANT_CALLING_OUTPUT_PATH="[YOUR_OUTPUT_PATH]"         # e.g. chr21.vcf (please make sure the directory exists)
CONTIG_NAME="[CONTIG_NAME_FOR_VARIANT_CALLING]"          # e.g. chr21

python $CLAIR callVarBam \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bed_fn "$BED_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--call_fn "$VARIANT_CALLING_OUTPUT_PATH" \
--pypy "$PYPY" \
--ctgName "$CONTIG_NAME"

cd "$VARIANT_CALLING_OUTPUT_PATH"
```

#### Note
* In practice, we suggest you to use `callVarBamParallel` to generate multiple commands that invokes `callVarBam` on smaller chromosome chucks, instead of directly using `callVarBam` on a whole chromosome.
* You may consider using the `--pysam_for_all_indel_bases` option for more accurate results. On Illumina data and PacBio CCS data, the option requires 20% to 50% much running time. On ONT data, Clair can run two times slower, while the improvement in accuracy is not significant.
* About seeting an appropriate allele frequency cutoff, please refer to [About Setting the Alternative Allele Frequency Cutoff](#about-setting-the-alternative-allele-frequency-cutoff)

### Call variants from BAM in parallel (using `callVarBamParallel`)
```bash
# variables
SAMPLE_NAME="NA12878"
OUTPUT_PREFIX="var"

# create command.sh for run jobs in parallel
python $CLAIR callVarBamParallel \
--chkpnt_fn "$MODEL" \
--ref_fn "$REFERENCE_FASTA_FILE_PATH" \
--bed_fn "$BED_FILE_PATH" \
--bam_fn "$BAM_FILE_PATH" \
--pypy "$PYPY" \
--sampleName "$SAMPLE_NAME" \
--output_prefix $OUTPUT_PREFIX > command.sh

# disable GPU if you have one installed
export CUDA_VISIBLE_DEVICES=""

# run Clair with 4 concurrencies
cat command.sh | parallel -j4

# concatenate vcf files and sort the variants called
vcfcat var*.vcf | vcfstreamsort | bgziptabix snp_and_indel.vcf.gz
```

#### Note
* `callVarBamParallel` submodule generates `callVarBam` commands that can be run in parallel
* `parallel -j4` will run four concurrencies in parallel using GNU parallel. We suggest using half the number of available CPU cores (not threads).
* If [GNU parallel](https://www.gnu.org/software/parallel/) is not installed, please try ```awk '{print "\""$0"\""}' commands.sh | xargs -P4 -L1 sh -c```
* If no BED file was provided, Clair will call variants on the whole genome.
* `vcfcat`, `vcfstreamsort` and `bgziptabix` commands are from [vcflib](https://github.com/vcflib/vcflib).
* `CUDA_VISIBLE_DEVICES=""` makes GPUs invisible to Clair so it will use CPU for variant calling. Please notice that unless you want to run `commands.sh` in serial, you cannot use GPU because one running copy of Clair will occupy all available memory of a GPU. While the bottleneck of `callVarBam` is at the `CreateTensor` script, which runs on CPU, the effect of GPU accelerate is insignificant (roughly about 15% faster). But if you have multiple GPU cards in your system, and you want to utilize them in variant calling, you may want split the `commands.sh` in to parts, and run the parts by firstly `export CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the ID of the GPU to be used.
* If you are going to call on non-human BAM file (e.g. bacteria), add `--includingAllContigs` option to call on contigs besides chromosome 1-22/X/Y/M/MT
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
illumina | Illumina | HG001,2,3,4,5 | Novoalign | [Download](http://www.bio8.cs.hku.hk/clair_models/illumina/trained_models.tar)
pacbio/ccs | PacBio CCS | HG001,5 | Minimap2 | [Download](http://www.bio8.cs.hku.hk/clair_models/pacbio/ccs/trained_models.tar)
ont | ONT R9.4.1 | HG001,2 | Minimap2 | [Download](http://www.bio8.cs.hku.hk/clair_models/ont/trained_models.tar)

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
./pypy3 -m pip install -U pip wheel intervaltree
# Use pypy3 as an inplace substitution of python to run pypy-able scripts
```

To guarantee a good user experience (good speed), pypy must be installed to run `callVarBam` (call variants from BAM), and `callVarBamParallel` that generate parallelizable commands to run `callVarBam`.
Tensorflow is optimized using Cython thus not compatible with `pypy3`. For the list of scripts compatible to `pypy3`, please refer to the [Submodule Descriptions](#submodule-descriptions).

*Pypy is an awesome Python JIT intepreter, you can donate to [the project](https://pypy.org).*


---

## [TODO] Model Training
