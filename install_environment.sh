conda create -n rsc-net python=3.7
conda activate rsc-net

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install neural-renderer-pytorch==1.1.3
pip install opencv-python
pip install pyopengl==3.1.0
pip install pyrender==0.1.45
pip install tensorboard
pip install chumpy==0.70
pip install smplx==0.1.13
pip install spacepy==0.2.2
pip install tqdm
pip install trimesh
pip install scipy==1.2.1


