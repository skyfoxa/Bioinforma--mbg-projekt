# Bioinforma

## Master

Hlavna vetva - najaktualnejsi kod, **Master2** vetva je backup.

* na stiahnutie dat sa pouziva nastroj [Ferret](http://limousophie35.github.io/Ferret/)

Aplikácia obsahuje rozhranie pre spúšťanie rôznych štatistických testov. Momentálne sú naimplementované 2 druhy
testov:

* [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
* [Permutačný test](https://en.wikipedia.org/wiki/Resampling_(statistics)) - zpermutuje dáta a vykoná ľubovolný definovaný
        štatistický test

## Master2
### Spustenie

* python main.py -dataPath={path_to_data_folder} -fetch=True
	* Priznak fetch spusti stahovanie dat do zlozky urcenej cestou v **-dataPath**
	* V pripade ze priznak fetch je nastaveny na True, dojde k zmazaniu celej zlozky
	  urcenej cestou v **-dataPath**

* python main.py -dataPath={path_to_data_folder}
	* Ak priznak **-fetch** nie je definovany, pouziju sa data v zlozke urcenej cestou 
	  **-dataPath**
	* Data zostavaju zachoane, nic sa nemaze.

### Dependencies

#### Clustalw2

Pre vypocet multiple sequence alignment je potrebne mat nainstalovany program 
[Clustalw2](http://clustal.org/download/current/). Ten je potrebne umiestnit do zlozky **src/clustalw** pod nazvom **clustalw**

## Jokes

I don't trust atoms... I heard they make up everything.

This readme contains chemistry jokes, but you may not react
