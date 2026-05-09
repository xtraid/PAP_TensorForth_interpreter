#!/bin/bash
BINARY="./TensorForth"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    local expect="$2"   # "pass" o "fail"
    local content="$3"
    local expected_output="$4"  # opzionale: stringa attesa in stdout

    local tmpfile
    tmpfile=$(mktemp /tmp/tf_test_XXXXXX.tensorforth)
    printf '%s' "$content" > "$tmpfile"

    output=$("$BINARY" "$tmpfile" 2>&1)
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
echo "=================================================================="
echo "Risultato: ${PASS} PASS, ${FAIL} FAIL su $((PASS+FAIL)) test totali"
[ $FAIL -eq 0 ] && exit 0 || exit 1
