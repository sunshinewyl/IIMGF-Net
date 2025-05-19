# IIMGF-Net
Indepth Integration of Multi-Granularity Features Network

# Enviroments
1. Basic environment configuration<br><br>python 3.9, pytorch 2.1.1, torchvision 0.16.1<br><br>Other basic packages can be run directly
    
        'pip install -r requirements.txt'
2. Install the package related to Mamba<br><br>Download mamba 2.0.4 from [github](https://github.com/state-spaces/mamba/releases/tag/v2.0.4) and run:

        'pip install your/download/mamba/path'

        'pip install -e mamba-1p1p1'
3. Download causal_conv1d 1.4.0 from [github](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0) and run:

         'pip install your/download/causal_con1d/path'


# Data Preparation
Please at first download datasets [Derm7pt](https://derm.cs.sfu.ca/Download.html) and then download the pretrained model of 
VMamba-Tiny on ImageNet-1k from [github](https://github.com/MzeroMiko/VMamba). Replace their actual storage paths in the config file.

     
# Training & Testing

Train model IIMGF-Net, please run:


    `python main.py --config ./config/train.yaml`

Test model IIMGF-Net, please run:

    `python test.py --config ./config/test.yaml`
     

# Contact
For any questions, feel free to contact: `wuyli29@mail2.sysu.edu.cn`