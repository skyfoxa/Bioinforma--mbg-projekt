# Bioinformatika - Gene Correlations

## Popis

Git repozitar obsahuje 2 vetvy:
- [Master](https://github.com/skyfoxa/Bioinforma--mbg-projekt/tree/master)
  - Aktualna verzia skriptu. 
- [Master2](https://github.com/skyfoxa/Bioinforma--mbg-projekt/tree/master2)
  - Zaloha stareho kodu - obsahuje FTP klienta, ktory sa pripaja na [databazu 1000 genomes](http://www.internationalgenome.org/data). Tento pristup bol presunuty do zalohy, lebo data sa daju ziskat pohodlnejsim sposobom pomocou nastroja [Ferret](http://limousophie35.github.io/Ferret/)

Skript je napisany v jazyku **Python 3.6**. Obsahuje rozhranie pre spúšťanie rôznych štatistických testov. Momentálne sú naimplementované 2 druhy testov:

* [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
* [Permutačný test](https://en.wikipedia.org/wiki/Resampling_(statistics))
   - zpermutuje dáta a vykoná ľubovolný definovaný štatistický test

## Master

### Spustenie

#### Data

Pred samotnym spustenim je mozne stiahnut pozadovane data genov pouzitim nastroja [Ferret](http://limousophie35.github.io/Ferret/).

Zlozka **./Ferret** obsahuje data niekolkych genov, ktore je mozne pouzit ako vstup skriptu. Ako vstup sa vyuziva vzdy `.ped` subor. V zlozke **./Ferret/Negative** sa nachadzaju data pouzivane ako negativne kontroly.

#### Spustenie skriptu

Pre spustenie je potrebne mat nainstalovany aspon [Python 3.6](https://www.python.org/downloads/release/python-360/). 

##### Terminal (Linux)

V terminaly je potrebne spustit:
1.  `cd PROJECT_ROOT_FOLDER/src/`
2. `python3 ./main.py -gene1=./Ferret/eIF4E1/eIF4E1.ped -gene2=./Ferret/eIF4G1/eIF4G1.ped`

##### Terminal v GitBash/CMD (Windows)

Pre Windows je mozne pouzit nastroj [GitBash](https://git-scm.com/download/win) alebo klasicky Windows CMD
1. `cd PROJECT_ROOT_FOLDER/src/`
2. `python ./main.py -gene1=./Ferret/eIF4E1/eIF4E1.ped -gene2=./Ferret/eIF4G1/eIF4G1.ped`

##### PyCharm

Je mozne vyuzit [PyCharm IDE](https://www.jetbrains.com/pycharm/), ktory pracuje priamo s Pythonem. Program je zadarmo pre studentov(mozno aj ucitelov).
1. `File->Open` pre otvorenie `PROJECT_ROOT_FOLDER`
2. V `Run->Edit Configurations` v casti `Script parameters` sa nastavuju parametre vstupu
    - Napr.: `-gene1=../Ferret/eIF4E1/eIF4E1.ped -gene2=../Ferret/eIF4G1/eIF4G1.ped`
3. `Run->Run...`

#### Vystup

Vystup sa vypisuje do zlozky `PROJECT_ROOT_FOLDER/output/`.

Skript ma 3 druhy vystupu:
1. `Gene_1.ped - Gene_2.ped.png` subor
    - Obsahuje tabulku s poctom pozitivnych/negativnych zhod a ich pomer pre jednotlive testy
    - Obsahuje histogramy a best fit line pre jednotlive testy
2. Vystup do kozole
    - Obsahuje pocty pozitivnych/negativnych zhod a ich pomer
3. `Gene_1.ped - Gene_2.ped.txt` subor
     - Obsahuje popis korelaci medzi stlpcami Genu 1 s Genom 2 a naopak.

#### Automatizovane spustanie

V `PROJECT_ROOT_FOLDER` je subor `tests.sh`, ktory sluzi na automatizovane testovanie viacerych genov za sebou. Je mozne si upravit/dopisat vlastne testy. Hromadny vystup sa vypisuje jak do konzole tak do samostatneho suboru  **test.txt**. 

Skript je spustitelny cez:
- Linux/MAC OS: v termanili spustit `bash ./tests.sh`
- Windows: v GitBash spustit `bash ./tests.sh` alebo len `./tests.sh`

## Master2
### Spustenie

* python main.py -dataPath={path_to_data_folder} -fetch=True
	* Priznak fetch spusti stahovanie dat do zlozky urcenej cestou v **-dataPath**
	* V pripade ze priznak fetch je nastaveny na True, dojde k zmazaniu celej zlozky
	  urcenej cestou v **-dataPath**

* python main.py -dataPath={path_to_data_folder}
	* Ak priznak **-fetch** nie je definovany, pouziju sa data v zlozke urcenej cestou 
	  **-dataPath**
	* Data zostavaju zachovane, nic sa nemaze.

### Dependencies

#### Clustalw2

Pre vypocet multiple sequence alignment je potrebne mat nainstalovany program 
[Clustalw2](http://clustal.org/download/current/). Ten je potrebne umiestnit do zlozky **src/clustalw** pod nazvom **clustalw**

## Jokes

I don't trust atoms... I heard they make up everything.

This readme contains chemistry jokes, but you may not react
