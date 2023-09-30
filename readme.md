- Create a virtual environment 
```bash
conda create -n pneusam python=3.10 -y
conda activate pneusam
```
- Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
- install setup
```bash
pip install -e .
```

- if you don't have sam_ckpt folder, please download pre-train
```bash
bash downloadPre-train.sh
```
- To prepare data, please download data from [Dataset](https://drive.google.com/u/0/uc?id=10iG8XqtNeAfitYxnELfBXpW1ZBHtcYac&export=download) and run file convert2png.py

- To train and test please read more in train.py
```bash
python train.py
```

- To test and using demo, please down file and unzip [experiment](https://drive.google.com/file/d/1Ky24fXYulqKZpCDk0Ph4tWOAZwID7S4_/view)

- To using demo, please install PyQt5:
```bash
pip install PyQt5
```
or
```bash
conda install -c anaconda pyqt -y
```
Then run
```bash
python gui.py
```