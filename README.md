iAR package
===========

Description
===========

Data sets, functions and scripts with examples to implement
autoregressive models for irregularly observed time series. The models
available in this package are the irregular autoregressive model
[(Eyheramendy et al.(2018))](#1), the complex irregular autoregressive model
[(Elorrieta et al.(2019))](#2) and the bivariate irregular autoregressive model [(Elorrieta et al.(2021))](#3).

Contents
========

-   Irregular Autoregressive (IAR) Model [[1]](#1)
-   Complex Irregular Autoregressive (CIAR) Model [[2]](#2)
-   Bivariate Irregular Autoregressive (BIAR) Model [[3]](#3)

Instalation
=====================

Dependencies:

```
numpy
pandas
scipy
matplotlib
sklearn
statsmodels
```

Install from PyPI using:

```
pip install iar
```

or clone this github and do:

```
python setup.py install --user
```

Examples
======================

-   IAR Model demo [here](https://github.com/felipeelorrieta/iAR/blob/master/examples/IAR_demo.ipynb)
-   CIAR Model demo [here](https://github.com/felipeelorrieta/iAR/blob/master/examples/CIAR_demo.ipynb)
-   BIAR Model demo [here](https://github.com/felipeelorrieta/iAR/blob/master/examples/BIAR_demo.ipynb)

Authors
======================

-   Felipe Elorrieta (felipe.elorrieta@usach.cl) (Millennium Institute of Astrophysics and Universidad de Santiago de Chile)
-   Cesar Ojeda (Universidad del Valle - Colombia)
-   Susana Eyheramendy (Millennium Institute of Astrophysics and Universidad Adolfo Ibañez)
-   Wilfredo Palma (Millennium Institute of Astrophysics)

Acknowledgments
======================

The authors acknowledge support from the ANID – Millennium Science Initiative Program – ICN12_009 awarded to the Millennium Institute of Astrophysics MAS (www.astrofisicamas.cl) 

References
======================

<a id="1">[1]</a> Eyheramendy S, Elorrieta F, Palma W (2018). “An irregular discrete time series model to identify residuals with autocorrelation in astronomical light curves.” Monthly Notices of the Royal Astronomical Society, 481(4), 4311–4322. ISSN 0035-8711, doi: 10.1093/mnras/sty2487, https://academic.oup.com/mnras/article-pdf/481/4/4311/25906473/sty2487.pdf.

<a id="2">[2]</a> Elorrieta, F, Eyheramendy, S, Palma, W (2019). “Discrete-time autoregressive model for unequally spaced time-series observations.” A\& A, 627, A120. doi: 10.1051/00046361/201935560, https://doi.org/10.1051/0004-6361/201935560.

<a id="3">[3]</a> Elorrieta, F, Eyheramendy, S, Palma, W, Ojeda, C (2021).A novel bivariate autoregressive model for predicting and forecasting irregularly observed time series, Monthly Notices of the Royal Astronomical Society, 505 (1),1105–1116,https://doi.org/10.1093/mnras/stab1216

<a id="4">[4]</a> Jordán A, Espinoza N, Rabus M, Eyheramendy S, Sing DK, Désert J, Bakos GÁ, Fortney JJ, LópezMorales M, Maxted PFL, Triaud AHMJ, Szentgyorgyi A (2013). “A Ground-based Optical Transmission Spectrum of WASP-6b.” The Astrophysical Journal, 778, 184. doi: 10.1088/0004637X/
778/2/184, 1310.6048, https://doi.org/10.1088/0004-637X/778/2/184.

<a id="5">[5]</a> Lira P, Arévalo P, Uttley P, McHardy IMM, Videla L (2015). “Long-term monitoring of the archetype Seyfert galaxy MCG-6-30-15: X-ray, optical and near-IR variability of the corona, disc and torus.” Monthly Notices of the Royal Astronomical Society, 454(1), 368–379. ISSN 0035-8711, doi: 10.1093/mnras/stv1945, https://doi.org/10.1093/mnras/stv1945.

