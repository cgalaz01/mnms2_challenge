# Tempera: Spatial Transformer Feature Pyramid Network for Cardiac MRI Segmentation
This repository is the code for Tempera submitted to the M&amp;Ms-2 challenge: https://www.ub.edu/mnms-2/.
Details of the model can be found at: https://link.springer.com/chapter/10.1007/978-3-030-93722-5_29

## Setup
The code is implemented in Python and all libraries and their versions can be found in the file 'environment.yml'.

## Data
The data is publicly available and can be obtained from: https://www.ub.edu/mnms-2/.
The model expects the data to be located at:
```
mnms2_challenge/data/trainining
mnms2_challenge/data/validation
```
where training contains the samples from 1-160 and validation the samples from 161-200.

## Training the model
To train the model, simply run:
```
python src/run_training.py
```

## Inference
To make predictions using the trained model, first copy the trained weights of the model to:
```
src/model_weights/multi_stage_model/model.weights.h5
```
and run the inferenve script by:
```
python src/run_inference.py <input_path> <output_path>
```

## Citation
If you found this code useful for your project please cite as:
```
@inproceedings{galazis2021tempera,
  title={Tempera: Spatial transformer feature pyramid network for cardiac MRI segmentation},
  author={Galazis, Christoforos and Wu, Huiyi and Li, Zhuoyu and Petri, Camille and Bharath, Anil A and Varela, Marta},
  booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart},
  pages={268--276},
  year={2021},
  organization={Springer}
}
```
	
## Acknowledgement
This project was supported by the UK Research and Innovation (UKRI) Centres of Doctoral Training (CDT) in Artificial Intelligence for Healthcare (AI4H) http://ai4health.io (Grant No. EP/S023283/1) and the British Heart Foundation Centre of Research
Excellence at Imperial College London (RE/18/4/34215).
