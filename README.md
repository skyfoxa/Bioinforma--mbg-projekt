# Bioinforma

## Master

Hlavna vetva - najaktualnejsi kod, **Master2** vetva je backup.

* na stiahnutie dat sa pouziva nastroj [Ferret](http://limousophie35.github.io/Ferret/)

Na vypocet korelacie pouzivame metodu [ANOVA](https://cs.wikipedia.org/wiki/Anal%C3%BDza_rozptylu). Nullova
hypoteza je, ze su data zhodne. Ak je p-value <= 0.05, tak nullovu hypotezu zamietame.

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
