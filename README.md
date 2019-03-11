# Clair - Yet another deep neural network based variant caller  
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)  
Contact: Ruibang Luo  
Email: rbluo@cs.hku.hk  

***

## Introduction
__Clair__ is the successor of [Clairvoyante](https://github.com/aquaskyline/Clairvoyante). The usage of Clair is almost identical to Clairvoyante. Please refer to the [README](https://github.com/aquaskyline/Clairvoyante/blob/rbDev/README.md) in the Clairvoyante repo for more details.

## Usage
```shell
git clone https://github.com/HKU-BAL/Clair.git
cd Clair
./clair.py
```

## Models
Please download models from [here](http://www.bio8.cs.hku.hk/clair/models/).

Folder | Tech | Sample used | Depth used
--- | :---: | :---: | :---: |
pacbio/rsii | PacBio RSII | HG001,2,3,4 | "Full depth of each sample" x {0.1,0.2,0.4,0.6,0.8,1.0} |
pacbio/ccs | PacBio CCS 15k | HG002 | "Full depth" x {0.1,0.2 ..., 0.9} |
ont/r94 | ONT R9.4 (no flip-flop) | HG001 | "Full depth" x {0.1,0.2 ..., 0.9} |