# HDR imaging


## Requirement

- python3 (or higher)
- opencv 3.0 (or higher)

You will need to install some package using `pip3`:

- numpy
- matplotlib

## Usage

```bash
$ python HDR-playground.py <input img dir> <output .hdr name>

# for example
$ python ./HDR-playground.py taipei taipei.hdr
```

I also provide an jupyter version(`HDR-playground.ipynb`), it’s function is same as `HDR-playground.py`. jupyter is really convenient for develop python program!

## Input format

The input dir should have:

- Some `.png` images
- A `image_list.txt`, file should contain:
  - filename
  - exposure
  - 1/shutter_speed

This is an example for `image_list.txt`:

```
# Filename   exposure 1/shutter_speed
DSC_0058.png 32        0.03125
DSC_0059.png 16        0.0625
DSC_0060.png  8        0.125
DSC_0061.png  4        0.25
DSC_0062.png  2        0.5
DSC_0063.png  1        1
DSC_0064.png  0.5      2
DSC_0065.png  0.25     4
DSC_0066.png  0.125    8
DSC_0067.png  0.0625  16
```

## Output

The program will output:

- A `.hdr` image
- A reconstruct RGB response curve plot
- A pseudo-color radiance map(with log value)

for sample output, you can see [HDR-playground.ipynb](https://github.com/SSARCandy/HDR-imaging/blob/master/HDR-playground.ipynb) as reference.

## Tonemap

I use tmo for tonemapping, it implement 24 algorithms.  
I write a script `tonemap.bat` for auto-run all 24 algorithms. 

```bash
$ tonemap.bat <filename without extension>
```

Make sure all `tm_*.exe` is in your system PATH

## Environment

I test my code in Window10, but it should work fine in macOS/Linux(exclude tonemapping reference program need run in Windows)
