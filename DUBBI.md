# Dubbi da chiarire col prof

## I/O — `{` e `}`

1. **Formato file per `{`/`}`**: dal PDF sembrano usare il formato binario `on_disk_tensor`
   (header con `shape`, `ndim`, `data_offset=64`, poi float puri). Confermare che NON
   siano per PGM come `(`/`)`.

2. **Normalizzazione in `{`/`}`**: visto che il file contiene float grezzi, i valori
   vengono salvati/caricati as-is (nessun /255 o *255). Corretto?

3. **`data_offset` deve essere esattamente 64?** Con `MAX_DIM=2`:
   `sizeof(on_disk_tensor)` = 4+4+4+4 (padding) +8 = 24 bytes. Il PDF dice "minimo
   offset allineato a 64 byte", quindi `data_offset=64`. Confermare.

## I/O — `)`

4. **`)` senza filename**: se in cima allo stack c'è un tensore invece di una stringa,
   cosa deve fare `)` ? Errore? Salvare in un file di default? Il PDF dice sempre
   `( a filename -- )`, quindi dovrebbe essere un errore — ma nella nostra sessione
   si era discusso di stamparlo come immagine.

## Operatori mancanti (non implementati)

5. **`c` — convoluzione 2D**: il risultato deve avere la stessa dimensione di `a`
   (con zero-padding). Confermare stride=1 e padding simmetrico (floor(k/2) per lato).

6. **`_` — ravel**: non deve copiare i dati in memoria (come reshape). Confermare
   che basti cambiare `shape` nell'istanza senza toccare `data`.

7. **`#` — shape**: ritorna un tensore 1D con le dimensioni. Per un tensore 1D di N
   elementi ritorna `[ N ]` (1 elemento) o `[ N 1 ]` (2 elementi)?

8. **`?` — random**: usa `rand()` normalizzato in [0,1]. Nessun seed fisso richiesto?

9. **`f` — fill**: `[ 2 3 ] [ 1 2 ] f` crea la matrice `[[1 2 1],[2 1 2]]`. Come
   si comporta se il numero di elementi di `v` non divide il numero totale di celle?
   (es. `[ 2 3 ] [ 1 2 3 ] f` → 6 celle, 3 valori: funziona. `[ 2 3 ] [ 4 ] f` →
   tutti 4? Confermare.)

10. **`s` — swap**: il token `s` è già usato? No, nel dizionario attuale `S`
    (maiuscola) è la somma. `s` minuscola = swap è libero. Confermare.

11. **`m`/`M` — min/max**: token `m` e `M`. Attualmente non presenti. Confermare
    che non confliggano con altri token.

12. **`R` — relu**: token `R` maiuscola. Libero. Confermare.

## Stack e semantica

13. **`d` (dup) e `o` (over)**: il PDF dice di non copiare il tensore ma solo
    incrementare il ref_count. Già implementato così con `ref_count`. Confermare
    che drop (`D`) decrementi il ref_count e faccia free solo a zero.

14. **Errori fatali vs. warning**: quando c'è un errore (stack underflow, shape
    incompatibili, file non trovato) il programma deve uscire con codice ≠ 0.
    Attualmente ritorniamo -1 che diventa exit code 255. Va bene o serve un codice
    specifico?

## OpenMP

15. **Quali operazioni parallelizzare?**: il PDF suggerisce di parallelizzare
    operazioni element-wise (+, -, *, confronti, logiche, convoluzione). Confermare
    che `?` (random) NON vada parallelizzato (rand() non thread-safe).

16. **Convoluzione parallela**: il PDF dice esplicitamente che ogni output pixel è
    indipendente → parallelizzabile con `#pragma omp parallel for`. Confermare.

## Formato output di `p`

17. **Formato esatto di `p`**: il PDF dice `Tensor(shape=[shape], data=[data])`.
    Per un tensore 2D 3x2 la shape è `[3 2]` o `[3, 2]`? Nessuna virgola nel
    formato, confermare.
