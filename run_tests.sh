#!/bin/bash
BINARY="./TensorForth"
PASS=0
FAIL=0

# Uso: ./run_tests.sh [--valgrind]
VALGRIND=""
if [ "${1}" = "--valgrind" ]; then
    VALGRIND="valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=definite,indirect --quiet"
    echo ">>> Modalità valgrind attiva <<<"
fi

run_test() {
    local name="$1"
    local expect="$2"   # "pass" o "fail"
    local content="$3"
    local expected_output="$4"  # opzionale: stringa attesa in stdout

    local tmpfile
    tmpfile=$(mktemp /tmp/tf_test_XXXXXX.tensorforth)
    printf '%s' "$content" > "$tmpfile"

    output=$(${VALGRIND} "$BINARY" "$tmpfile" 2>&1)
    exit_code=$?
    rm -f "$tmpfile"

    local ok=1
    if [ "$expect" = "pass" ] && [ $exit_code -ne 0 ]; then ok=0; fi
    if [ "$expect" = "fail" ] && [ $exit_code -eq 0 ]; then ok=0; fi
    if [ -n "$expected_output" ] && ! echo "$output" | grep -qF -- "$expected_output"; then ok=0; fi

    if [ $ok -eq 1 ]; then
        printf "PASS  %s\n" "$name"
        PASS=$((PASS+1))
    else
        printf "FAIL  %s\n" "$name"
        printf "      atteso=%s  exit=%d\n" "$expect" "$exit_code"
        [ -n "$output" ] && printf "      output: %s\n" "$output"
        [ -n "$expected_output" ] && printf "      stringa attesa: %s\n" "$expected_output"
        FAIL=$((FAIL+1))
    fi
}

echo "=== TEST VALIDI ==================================================="

run_test "push e print base" \
    pass \
    "[ 1.0 2.0 3.0 ] p" \
    "Tensor(shape=[1 3]"

run_test "addizione element-wise" \
    pass \
    "[ 1.0 2.0 3.0 ] [ 4.0 5.0 6.0 ] + p" \
    "5.000000 7.000000 9.000000"

run_test "sottrazione element-wise (TOS - second)" \
    pass \
    "[ 5.0 3.0 1.0 ] [ 1.0 1.0 1.0 ] - p" \
    "-4.000000 -2.000000 0.000000"

run_test "moltiplicazione element-wise" \
    pass \
    "[ 2.0 3.0 4.0 ] [ 2.0 2.0 2.0 ] * p" \
    "4.000000 6.000000 8.000000"

run_test "reshape a matrice 2x2" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r P" \
    ""

run_test "matmul identita 2x2" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r [ 1.0 0.0 0.0 1.0 ] [ 2.0 2.0 ] r @ p" \
    "1.000000 2.000000 3.000000 4.000000"

run_test "matmul 2x2 generica (TOS @ second)" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r [ 5.0 6.0 7.0 8.0 ] [ 2.0 2.0 ] r @ p" \
    "23.000000 34.000000 31.000000 46.000000"

run_test "somma riduzione" \
    pass \
    "[ 1.0 2.0 3.0 4.0 5.0 ] S p" \
    "15.000000"

run_test "dot product" \
    pass \
    "[ 1.0 2.0 3.0 ] [ 4.0 5.0 6.0 ] . p" \
    "32.000000"

run_test "duplica e somma" \
    pass \
    "[ 1.0 2.0 3.0 ] d + p" \
    "2.000000 4.000000 6.000000"

run_test "confronto maggiore (TOS > second)" \
    pass \
    "[ 1.0 3.0 2.0 ] [ 2.0 2.0 2.0 ] > p" \
    "1.000000 0.000000 0.000000"

run_test "confronto minore (TOS < second)" \
    pass \
    "[ 1.0 3.0 2.0 ] [ 2.0 2.0 2.0 ] < p" \
    "0.000000 1.000000 0.000000"

run_test "uguaglianza" \
    pass \
    "[ 1.0 2.0 3.0 ] [ 1.0 9.0 3.0 ] = p" \
    "1.000000 0.000000 1.000000"

run_test "not logico" \
    pass \
    "[ 1.0 0.0 1.0 0.0 ] ! p" \
    "0.000000 1.000000 0.000000 1.000000"

run_test "and logico" \
    pass \
    "[ 1.0 0.0 1.0 0.0 ] [ 1.0 1.0 0.0 0.0 ] & p" \
    "1.000000 0.000000 0.000000 0.000000"

run_test "or logico" \
    pass \
    "[ 1.0 0.0 1.0 0.0 ] [ 1.0 1.0 0.0 0.0 ] | p" \
    "1.000000 1.000000 1.000000 0.000000"

run_test "mask (seleziona da due tensori)" \
    pass \
    "[ 0.0 0.0 0.0 ] [ 10.0 20.0 30.0 ] [ 1.0 0.0 1.0 ] \$ p" \
    ""

run_test "array singolo elemento" \
    pass \
    "[ 42.0 ] p" \
    "42.000000"

run_test "reshape 1D (shape a 1 elemento)" \
    pass \
    "[ 1.0 2.0 3.0 ] [ 3.0 ] r p" \
    "Tensor(shape=[1 3]"

run_test "reshape 1x6 a 2x3" \
    pass \
    "[ 1.0 2.0 3.0 4.0 5.0 6.0 ] [ 2.0 3.0 ] r P" \
    ""

run_test "negativi e float" \
    pass \
    "[ -1.5 2.5 -3.5 ] [ 1.5 1.5 1.5 ] + p" \
    "0.000000 4.000000 -2.000000"

echo ""
echo "=== TEST ERRORI SINTASSI ARRAY ===================================="

run_test "manca spazio dopo [" \
    fail \
    "[1.0 2.0 ]" \
    "manca lo spazio obbligatorio dopo"

run_test "manca spazio prima di ]" \
    fail \
    "[ 1.0 2.0]" \
    "manca lo spazio obbligatorio prima"

run_test "spazio doppio tra elementi" \
    fail \
    "[ 1.0  2.0 ]" \
    "spazio doppio"

run_test "token non valido (lettera)" \
    fail \
    "[ 1.0 abc ]" \
    "token non valido"

run_test "manca chiusura ]" \
    fail \
    "[ 1.0 2.0 " \
    "manca la chiusura"

run_test "token non valido subito dopo [" \
    fail \
    "[ abc ]" \
    "token non valido"

run_test "token non valido in mezzo" \
    fail \
    "[ 1.0 2.0 @ 3.0 ]" \
    "token non valido"

echo ""
echo "=== TEST ERRORI OPERAZIONI ========================================"

run_test "matmul con vettore (TOS e' vettore)" \
    fail \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r [ 1.0 2.0 ] @" \
    "2D"

run_test "matmul con vettore (entrambi vettori)" \
    fail \
    "[ 1.0 2.0 ] [ 3.0 4.0 ] @" \
    "2D"

run_test "matmul shape incompatibili (3 cols vs 2 rows)" \
    fail \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r [ 1.0 2.0 3.0 4.0 5.0 6.0 ] [ 2.0 3.0 ] r @" \
    "incompatibili"

run_test "reshape prodotto incompatibile" \
    fail \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 3.0 ] r" \
    "incompatibile"

run_test "reshape shape con piu di 2 elementi" \
    fail \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 1.0 ] r" \
    ""

run_test "not su array non booleano" \
    fail \
    "[ 0.5 1.0 0.0 ] !" \
    "booleano"

run_test "and su array non booleano" \
    fail \
    "[ 0.5 1.0 ] [ 1.0 0.0 ] &" \
    "operazione logica"

run_test "mask su maschera non booleana" \
    fail \
    "[ 0.0 0.0 ] [ 1.0 2.0 ] [ 0.5 0.5 ] \$" \
    "booleano"

run_test "stack underflow su print" \
    fail \
    "p" \
    ""

run_test "stack underflow su operazione binaria" \
    fail \
    "[ 1.0 2.0 ] +" \
    ""

run_test "shape incompatibili su addizione" \
    fail \
    "[ 1.0 2.0 ] [ 1.0 2.0 3.0 ] +" \
    "incompatibili"

run_test "dot product su matrici 2D" \
    fail \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r [ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r ." \
    ""

echo ""
echo "=== TEST SWAP / OVER / DROP ======================================="

run_test "swap inverte TOS e second" \
    pass \
    "[ 1.0 ] [ 2.0 ] s p p" \
    "1.000000"

run_test "over duplica second" \
    pass \
    "[ 1.0 ] [ 2.0 ] o p p p" \
    "1.000000"

run_test "drop elimina TOS" \
    pass \
    "[ 1.0 ] [ 2.0 ] D p" \
    "1.000000"

run_test "drop underflow" \
    fail \
    "D" \
    ""

echo ""
echo "=== TEST RAVEL / SHAPE ==========================================="

run_test "ravel 2D -> 1D" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r _ p" \
    "Tensor(shape=[1 4]"

run_test "ravel 1D rimane 1D" \
    pass \
    "[ 1.0 2.0 3.0 ] _ p" \
    "Tensor(shape=[1 3]"

run_test "shape di 1D" \
    pass \
    "[ 5.0 6.0 7.0 ] # p" \
    "1.000000 3.000000"

run_test "shape di 2D" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r # p" \
    "2.000000 2.000000"

echo ""
echo "=== TEST FILL / RANDOM ==========================================="

run_test "fill crea matrice ciclica" \
    pass \
    "[ 2.0 3.0 ] [ 1.0 2.0 ] f p" \
    "1.000000 2.000000 1.000000 2.000000 1.000000 2.000000"

run_test "fill 1D" \
    pass \
    "[ 4.0 ] [ 7.0 ] f p" \
    "7.000000 7.000000 7.000000 7.000000"

run_test "random shape 1D" \
    pass \
    "[ 5.0 ] ? p" \
    "Tensor(shape=[1 5]"

run_test "random shape 2D" \
    pass \
    "[ 3.0 4.0 ] ? p" \
    "Tensor(shape=[3 4]"

echo ""
echo "=== TEST RELU / MIN / MAX ========================================="

run_test "relu azzera negativi" \
    pass \
    "[ -1.0 2.0 -3.0 4.0 ] R p" \
    "0.000000 2.000000 0.000000 4.000000"

run_test "relu tutti positivi inalterati" \
    pass \
    "[ 1.0 2.0 3.0 ] R p" \
    "1.000000 2.000000 3.000000"

run_test "min element-wise" \
    pass \
    "[ 1.0 5.0 3.0 ] [ 4.0 2.0 3.0 ] m p" \
    "1.000000 2.000000 3.000000"

run_test "max element-wise" \
    pass \
    "[ 1.0 5.0 3.0 ] [ 4.0 2.0 3.0 ] M p" \
    "4.000000 5.000000 3.000000"

echo ""
echo "=== TEST I/O DISCO (save + load roundtrip) ======================="

TMPBIN=$(mktemp /tmp/tf_tensor_XXXXXX.bin)

run_test "save tensore su disco" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] \"$TMPBIN\" }" \
    ""

run_test "load tensore da disco" \
    pass \
    "\"$TMPBIN\" { p" \
    "1.000000 2.000000 3.000000 4.000000"

run_test "roundtrip save+load 2D" \
    pass \
    "[ 1.0 2.0 3.0 4.0 ] [ 2.0 2.0 ] r \"$TMPBIN\" } \"$TMPBIN\" { p" \
    "Tensor(shape=[2 2]"

rm -f "$TMPBIN"

run_test "load file inesistente" \
    fail \
    "\"/tmp/nonexistent_tf_xyz.bin\" {" \
    ""

echo ""
echo "=== STRESS TEST =================================================="

run_test "stress: matmul 100x100" \
    pass \
    "[ 100.0 100.0 ] ? [ 100.0 100.0 ] ? @ D" \
    ""

run_test "stress: 1000 dup+drop" \
    pass \
    "$(python3 -c "print('[ 1.0 ] ' + ' '.join(['d D'] * 1000) + ' D')")" \
    ""

run_test "stress: fill grande e sum" \
    pass \
    "[ 1000.0 1000.0 ] [ 1.0 ] f S p" \
    "1000000.000000"

run_test "stress: chain operazioni" \
    pass \
    "[ 100.0 ] ? [ 100.0 ] ? + [ 100.0 ] ? - [ 100.0 ] ? * S D" \
    ""

run_test "stress: save+load roundtrip grande" \
    pass \
    "$(TMPF=$(mktemp /tmp/tf_stress_XXXXXX.bin); echo "[ 1000.0 ] ? \"$TMPF\" } \"$TMPF\" { S p"; rm -f "$TMPF")" \
    ""

echo ""
echo "=================================================================="
echo "Risultato: ${PASS} PASS, ${FAIL} FAIL su $((PASS+FAIL)) test totali"
[ $FAIL -eq 0 ] && exit 0 || exit 1
