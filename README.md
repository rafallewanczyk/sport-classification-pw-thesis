# Rozpoznawanie dyscypliny sportu na podstawie materiałów wideo 

Rozpoznawanie dyscypliny sportu jest zadaniem klasyfikacji aktywności grupowych. Ma ono zastosowanie przy organizowaniu dużych zbiorów nagrań wideo w celu np. proponowanie treści, lub wyszukiwaniu po słowach kluczowych. 

W tym projekcie przetestowane zostały podejścia do klasyfikacji aktywności grupowych poprzez analizę aktywności pojedynczych osób przy pomocy cech RGB lub cech szkieletowych, analizę cech RGB całych klatek nagrania oraz kombinacji tych metod. Opisane zostały szczegółowe architektury modeli oraz wnioski z ich działania. Dokonane zostało również porównanie z modelami bazowymi na badanym zbiorze. 

Efektem pracy jest model osiągający skuteczność wynoszącą 80.56% na zbiorze danych SVW oraz cały system przetwarzania wideo ekstrahujący wymagane cechy. 

## Struktura projektu
* `dataset` - moduł wczytujący zbiór danych
* `model` - implementacja modeli uczenia maszynowego w bibliotece Keras 
  * `train_notebooks/train_model` - notebook trenujący model oraz analizujący jego skuteczność
* `preprocessing` - moduł implementujący wstępne przetworzenie danych tj. detekcja osób, detekcja cech szkieletowych, śledzenie osób
* `utils` - wizualizacja cech
* `docs` - praca opisująca szczegółowo działanie projektu oraz rezultaty 


