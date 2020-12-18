# Mexico-Newspaper-Articles-Prod![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)
This repository contains the code to replicate the analysis in Erlich et al. 2021. Please see the article for a fuller description of the articles downloaded. 


## Description
This dataset contains newspaper articles from two news publications: 
- Reforma (219,354 articles)
- El Universal (119,302 articles)


Articles gathered are those that contain either:
- Name of national agency
- Abbreviation of agency
- Name of minister in charge of agency


## Compiled File description
Stable articles have been compiled into `all_unstable_articles.csv` (embargoed) with the following attributes.

![img](https://i.imgur.com/c6iyjmj.png)
- `source` : newspaper from which article was collected
- `date` : date of article publishing
- `title` : title of article
- `byline` : author(s) of article
- `section` : section of newspaper where article appears
- `length` : number of words in article
- `story` : article text
- `agency`: national agency mentioned in article

## Agency names
- `CFE` : Comisión Federal de Electricidad
- `INM` : Instituto Nacional de Migración
- `SEDENA` : Secretaría de la Defensa Nacional
- `SENER` : Secretaría de Energía
- `SSA` : Secretaría de Salud
- `COFEPRIS` : Comisión Federal para la Protección contra Riesgos Sanitarios
- `PEMEX` : Petróleos Mexicanos
- `SEDESOL` : Secretaría de Desarrollo Social
- `SEP` : Secretaría de Educación Pública
- `SSP` : Secretaría de Seguridad Pública
- `CONAGUA` : Comisión Nacional del Agua
- `PGR` : Procuraduría General de la República.
- `SEECO` : Secretaría de Economía
- `SFP` : Secretaría de la Función Pública
- `IMPI` : Instituto Mexicano de la Propiedad Industrial
- `SAGARPA` : Secretaría de Agricultura, Ganadería, Desarrollo Rural, Pesca
- `SEGOB` : Secretaría de Gobernación
- `SHCP` : Secretaría de Hacienda y Crédito Público
- `IMSS` : Instituto Mexicano del Seguro Social
- `SCT` : Secretaría de Comunicaciones y Transportes
- `SEMARNAT` : Secretaría de Medio Ambiente y Recursos Naturales
- `SRE` : Secretaría de Relaciones Exteriores

## Installation
Data is local. When you clone this repository, it will only contain references to data, not data itself.

Please ask the authors to share the data if you desire to run the replication on the same data. 

## Article Loading
```
Python:

import pandas as pd
df = pd.read_pickle('./data/raw/all_unstable_articles.pkl')
```

## Corruption related words
List one is specially curated based on local context and knowledge and involves some partially stemmed words.
### corruption_relevant_words
```
corrupción
corrupt
escándal
escandal
impunidad
soborn
irregularidad
indagatoria
incumpl
conflicto de intereses
fraud
indebid
malvers
complicidad
negligen
anomalía
desfalc
despilfarr
desví
enriquecimient
```

## Running Sequence (as of Jan 15, 2019)


## Contact
Benjamin Bagozzi - [bagozzib@gmail.com](mailto:bagozzib@gmail.com)

Daniel Berliner - [danberliner@gmail.com](mailto:danberliner@gmail.com)

Aaron Erlich - [aaron.erlich@mcgill.ca](mailto:aaron.erlich@mcgill.ca)


## License
[MIT](https://choosealicense.com/licenses/mit/)
