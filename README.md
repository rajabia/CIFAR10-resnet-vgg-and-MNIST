

TO Run the code:

1. Create a virtual enviroment: python3 -m venv myenv
2. Activate your virtual enviroment: source myenv/bin/activate.csh
3. Install pytorch and torchvision:   pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
4. Run the code with your parameters: python trainmodels.py --dataset cifar10 --epochs 100 --lr 0.03 --gamma 0.99
