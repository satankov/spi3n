# Spi3n < Spin NN >
Module to simulate spin models and use deep machine learning to make useful predictions and extract new knowledge.   



## Install package
Simple installation  
```commandline 
pip install .
```
    
    
Edit mode installation  
```commandline 
pip install -e .
```



## Navigation
### spi3n/
Use this as package. You can simply import and start developing
```python 
import spi3n
```


### NN/
This folder contains jupyter notebooks and examples of NN pipelines.  


- **phase_prediction/baseline/**  
This is the simpliest baseline example for phase prediction.  
It contains customizable notebook for training `baseline.ipynb` and `analyse_results.ipynb` to analyse NN outputs and extract critical exponent.  


### MC/
This folder contains scripts for data generation.  
Curently available scripts to generate `ising`, `bw` (Baxter-Wu) data.  

To generate data you need:  
- Set temperature range in file. Example, t_range.txt.  
- run script file  
```commandline 
python3 ising.py 12 t_range.txt 0 499
```
1st arguments is lattice size `L` (in example 12), 2nd is path to file with temperature points `t_path` (in example t_range.txt, separate points using comma), 3rd is index of temperature array `t_idx` (in example 0, it is made to be able to run in parallel), 4th is total amount of images to generate `N_img` (in example 499).  

The same logic can be applied for both models: Ising and Baxter-Wu.  



## Contact
Feel free to dm me:
- mail: satankow@yandex.ru
- tg: @satankov
