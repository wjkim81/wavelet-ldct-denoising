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

.../data/denoising/train/mayo/full_1mm
|                         /full_3mm
|                         /quarter_1mm
|                         /quarter_3mm
   /data/denoising/test/mayo/full_1mm
   |                         /full_3mm
|                         /quarter_1mm
|                         /quarter_3mm
|
## Running the code

* Training the model
```
python train.py --model <model> --datasets <list of data>
```
* Test the mode
```
python test.py --model <model> --test_datasets <list of data>
```

waveletdt is our propose model trained with $L_{wo}$ and waveletganp is our proposed model trained with $L_{wp}$