# Zdrojový kód k diplomové práci
### Téma práce: Využití kamerového systému pro zajištěni bezpečnosti osob na pracovišti
### Vypracoval: Filip Łuński LUN0024

## Instalace
- Pro spuštění skriptů vytvořte virtuální prostředí a nainstalujte závislosti uložené v requirements.txt

- Je třeba pamatovat, že pro využití grafické akcelerace je nutné zvlášť nainstalovat verzi PyTorch pro Vaší GPU.

## Struktura složek

- *dataset_creators* 
    - *datasetCreator.py* - skript pro tvoření trénovacích dat z videí.
- *pose_estimation_tests* - 
    - *pose_test.py* - skript pro testování různých detektorů pózy
- *fall_detection*
    - *KeypointClassifier<typ sítě>* - implementace neuronové sítě daného typu
    - *FallDetector.py* - implementace detektoru pádů
    - *training_model_<typ sítě>.py* - skripty pro trénování a testování modelů *KeypointClassifier*
    - *fall_finder_video.py* - skript pro detekci pádů ve videu využívající *FallDetector*
    - *FFNN.ckpt* a *GRU.ckpt* - uložené váhy modelů navržených v této práci

## Použití

Ve skriptech je třeba vždy zavolat metodu *main* s vhodnými parametry. Většinou je mj. třeba metodě předat správné cesty k souborům.

## Příkladové video

V souboru *cauca_fall_example_annotated.avi* se nachází příklad detekce pádu v jednom z videí z datasetu CAUCAFall. Pro detekci byl použit model GRU navržený v práci.