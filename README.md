# LDCT wavelet denoising

## Pytorch Wavelets

We implemented stationary wavelet transform with the help of pytorch wavelets.
You need to first install pytorch_wavelets to run wavelet transform.

```
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```


## Mayo Clinic Dataset
You need save Mayo Clinic dataset properly.
By default, we located our project repository and mayo clinic dataset as follows:

data <br/>
├── denoising <br/>
│   ├── train <br/>
│   │   └── mayo <br/>
│   │       ├── full_1mm <br/>
│   │       ├── full_3mm <br/>
│   │       ├── quarter_1mm <br/>
│   │       └── quarter_3mm <br/>
│   └── test <br/>
│       └── mayo <br/>
│           ├── full_1mm <br/>
│           ├── full_3mm <br/>
│           ├── quarter_1mm <br/>
│           └── quarter_3mm <br/>
works ── wavelet-ldct-denoising <br/>

## Running the code

* Training the model
```
python train.py --model <model> --datasets <list of data>
```
* Test the mode
```
python test.py --model <model> --test_datasets <list of data>
```

waveletdt is our proposed model trained with $L_{wo}$ and waveletganp is our proposed model trained with $L_{wp}$