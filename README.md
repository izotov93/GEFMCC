# Global Efficiency of entropy calculated using Matthews Correlation Coefficient (GEFMCC)

Classification of time series using machine learning (ML) 
analysis and entropy-based features is an urgent task for 
studying nonlinear signals in the fields of finance, biology 
and medicine, in-cluding EEG analysis and Brain-Computer 
Interfacing. As several entropy measures exist, the problems 
is assessing the effectiveness of entropies used as features 
for ML classification of non-linear dynamics of time series. 
We propose a method for assessing the effectiveness of entropy 
features using several chaotic mappings. We analyze fuzzy 
entropy (FuzzyEn) and neural net-work entropy (NNetEn) on four 
discrete mappings: the logistic map, the sine map, the Planck 
map, and the two-memristor-based map, with a base length time 
series of 300 elements. FuzzyEn has greater global efficiency 
(GEFMCC) in the classification task compared to NNetEn. 
However, NNetEn classification efficiency is higher than 
FuzzyEn for some local areas of the time series dynamics. 
The results of using horizontal visibility graphs (HVG) 
instead of the raw time series demonstrate the GEFMCC decrease
after HVG time series transformation. However,  the classification 
efficiency increases after applying the HVG for some local areas of 
the time se-ries dynamics. The scientific community can use the 
results to explore the efficiency of entropy-based classification 
of time series. An implementation of the algorithms in Python is 
presented.

## Citing the Work

[Link to article](https://www.preprints.org/manuscript/202312.2330 "preprints.org")

```shell
Conejero, J.A.; Velichko, A.; Garibo-i-Orts, Ò.; Izotov, Y.; Pham, V. 
Exploring Entropy-Based Classification of Time Series Using 
Visibility Graphs from Chaotic Maps. Preprints 2023, 
2023122330. https://doi.org/10.20944/preprints202312.2330.v1
```

## Installation

To install the package run the following command:
```shell
pip install -r requirements.txt
```

## How to use

The final script average_gefmcc.py initiates all the scripts,
for 4 different chaotic mappings, and calculates the average
entropy values using Equation. After the script is completed,
a file ‘average_GEFMCC.txt’ is created, which contains a 
string of GEFMCC values and the average GEFMCC value for 
all chaotic mappings, as an estimate of the efficiency of 
entropy.
```shell
python average_gefmcc.py
```
The calculation configuration is specified in the base_config dictionary in script.