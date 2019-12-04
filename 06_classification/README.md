### Instrukcja

1. Zakładamy, że nasze obrazy znajdują się w katalogu `image-dataset'.

    Przykładowa struktura katalogu roboczego:
    ```csv
    image-dataset\
        cls_1\
            img1.jpg
            img2.jpg
            ...
        cls_2\
            img1.jpg
            img2.jpg
            ...
    ```

2. Uruchomienie skryptu `01_prepare.py`. Skrypt utworzy nowy katalog w katalogu roboczym o nazwie `images`
oraz podzieli obrazy na zbiór treningowy, walidacyjny oraz testowy. 
    Struktura katalogu roboczego po uruchomieniu:
    ```csv
    image-dataset\
        images\
            cls_1\
                img1.jpg
                img2.jpg
                ...
            cls_2\
                img1.jpg
                img2.jpg
                ...
    images\
        train\
            cls_1\
               img1.jpg
               img3.jpg
               ...
            cls_2\
               img2.jpg
               img3.jpg
               ...
        valid\
            cls_1\
               img2.jpg
               img5.jpg
               ...
            cls_2\
               img1.jpg
               img4.jpg
               ...
        test\
            cls_1\
               img4.jpg
               img7.jpg
               ...
            cls_2\
               img5.jpg
               img8.jpg
               ...   
    ```    
3. Uruchomienie trenowania. Flaga `-e` oznacza liczbę epok trenowania modelu.
    ```
    $ python train.py -e 20
    ```
   
4. Ocena modelu na zbiorze testowym. Flaga `-d` oznacza ścieżkę do zbioru, flaga `-m` ścieżkę do modelu.
    ```
    $ python predict.py -d images\test -m output/model_28_11_2019_17_57.hdf5
    ```
5. Klasyfikacja pojedynczego obrazu:
    ```
    $ python classify.py -i horse\00000002.jpg -m output\model_28_11_2019_17_57.hdf5
    ```