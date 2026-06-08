#!/usr/bin/env bash
# =============================================================================
# hard_test.sh  --  Adversarial BLACK-BOX test suite for `tensorforth`.
#
# Derived EXCLUSIVELY from the project specification (project_C.pdf).
# No source file and no existing test was inspected: this script treats the
# interpreter as an opaque binary and only relies on the contracts the spec
# guarantees:
#
#   R1. "il programma deve segnalare l'errore e uscire con un codice di errore
#        diverso da 0 (quindi non avere segmentation fault)".
#   R2. stack errors / type errors / shape-incompatibility MUST be reported.
#   R3. valid programs from the spec MUST run.
#   R4. on-disk tensor format: struct{int32 shape[2]; int32 ndim; off_t off;}
#       with data aligned to 64 bytes (data_offset == 64).
#   R5. resources must be released (checked opportunistically with valgrind).
#
# HARD-FAIL conditions (all spec-grounded):
#   * ANY input that kills the process with a signal (rc >= 128)  -> FAIL.
#   * a clearly-erroneous program that exits 0                    -> FAIL.
#   * a clearly-valid program that exits != 0                     -> FAIL.
#   * a deterministic result that comes out wrong                 -> FAIL.
#
# Ambiguous-by-spec inputs are only checked for "does not crash".
# =============================================================================
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="$ROOT/tensorforth"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0
declare -a FAILED

r() { printf '\033[31m%s\033[0m' "$1"; }
g() { printf '\033[32m%s\033[0m' "$1"; }
y() { printf '\033[33m%s\033[0m' "$1"; }

note_pass(){ PASS=$((PASS+1)); printf '  %s %s\n' "$(g ok  )" "$1"; }
note_fail(){ FAIL=$((FAIL+1)); FAILED+=("$1"); printf '  %s %s\n' "$(r FAIL)" "$1"; }

# run_prog <program-text>  -> sets OUT (stdout+stderr) and RC
run_prog(){
  local f="$TMP/p.tf"
  printf '%s\n' "$1" > "$f"
  OUT="$(cd "$ROOT" && timeout 25 "$BIN" "$f" 2>&1)"; RC=$?
}

crashed(){ [ "$RC" -ge 128 ]; }   # killed by signal => spec violation

# A crash is ALWAYS a failure, whatever the test category expected.
guard_crash(){
  if crashed; then
    note_fail "CRASH rc=$RC ($1) :: <<$2>> :: ${OUT:0:160}"
    return 0   # handled
  fi
  return 1     # no crash
}

expect_err(){   # label, program  -- must exit non-zero, must not crash
  run_prog "$2"; guard_crash "$1" "$2" && return
  if [ "$RC" -ne 0 ]; then note_pass "$1  (rc=$RC)"
  else note_fail "$1 :: expected ERROR but rc=0 :: out=${OUT:0:120}"; fi
}

expect_ok(){    # label, program  -- must exit 0
  run_prog "$2"; guard_crash "$1" "$2" && return
  if [ "$RC" -eq 0 ]; then note_pass "$1"
  else note_fail "$1 :: expected OK but rc=$RC :: out=${OUT:0:120}"; fi
}

expect_nocrash(){ # label, program  -- spec ambiguous: only require survival
  run_prog "$2"; guard_crash "$1" "$2" && return
  note_pass "$1  (no crash, rc=$RC)"
}

expect_data(){  # label, program, normalized-substring  -- value oracle
  run_prog "$2"; guard_crash "$1" "$2" && return
  local norm; norm="$(printf '%s' "$OUT" | tr -d ' \t')"
  if [ "$RC" -eq 0 ] && printf '%s' "$norm" | grep -qF "$3"; then
    note_pass "$1"
  else
    note_fail "$1 :: want rc0 & '$3' :: rc=$RC out=${OUT:0:140}"
  fi
}

hr(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# =============================================================================
hr "0. Command-line interface contract"
# spec: invocation is `tensorforth [source file]`
OUT="$(cd "$ROOT" && timeout 10 "$BIN" 2>&1)"; RC=$?
if crashed; then note_fail "CRASH on no-arg invocation rc=$RC"
elif [ "$RC" -ne 0 ]; then note_pass "no source-file argument -> error (rc=$RC)"
else note_fail "no source-file argument should error, got rc=0"; fi

OUT="$(cd "$ROOT" && timeout 10 "$BIN" "$TMP/does_not_exist_$$.tf" 2>&1)"; RC=$?
if crashed; then note_fail "CRASH on missing source file rc=$RC"
elif [ "$RC" -ne 0 ]; then note_pass "missing source file -> error (rc=$RC)"
else note_fail "missing source file should error, got rc=0"; fi

# empty program == empty token sequence -> nothing to do, must succeed
printf '' > "$TMP/empty.tf"
OUT="$(cd "$ROOT" && timeout 10 "$BIN" "$TMP/empty.tf" 2>&1)"; RC=$?
if crashed; then note_fail "CRASH on empty program rc=$RC"
elif [ "$RC" -eq 0 ]; then note_pass "empty program -> rc=0"
else note_fail "empty program should be valid, rc=$RC"; fi

# whitespace-only program
printf '   \n\t  \n' > "$TMP/ws.tf"
OUT="$(cd "$ROOT" && timeout 10 "$BIN" "$TMP/ws.tf" 2>&1)"; RC=$?
if crashed; then note_fail "CRASH on whitespace program rc=$RC"
elif [ "$RC" -eq 0 ]; then note_pass "whitespace-only program -> rc=0"
else note_fail "whitespace-only program should be valid, rc=$RC"; fi

# =============================================================================
hr "1. Tensor literal lexing (spec: spaces significant, no commas)"
expect_err  "no inner spaces '[3.4 -8.5 3]'"      '[3.4 -8.5 3]'
expect_err  "commas '[ 3.4, -8.5, 3 ]'"           '[ 3.4, -8.5, 3 ]'
expect_err  "no space before ']'  '[ 1 2 3]'"     '[ 1 2 3]'
expect_err  "no space after '['   '[1 2 3 ]'"     '[1 2 3 ]'
expect_err  "unclosed bracket '[ 1 2 3'"          '[ 1 2 3'
expect_err  "stray close ']'"                     '1 2 3 ]'
expect_err  "lone ']'"                            ']'
expect_err  "operator inside literal '[ 1 + 2 ]'" '[ 1 + 2 ]'
expect_err  "non-number token '[ 1 x 3 ]'"        '[ 1 x 3 ]'
expect_err  "unterminated string"                 '"oops'
expect_err  "unknown token '%'"                   '%'
expect_err  "unknown word 'foo'"                  'foo'
expect_err  "second tensor unclosed"              '[ 1 2 ] [ 3 4'
expect_err  "nested bracket '[ [ 1 ] ]'"          '[ [ 1 ] ]'

# Ambiguous numeric forms: spec only says "floating point". Survival only.
expect_nocrash "trailing-dot '[ 1. 2. ]'"         '[ 1. 2. ]'
expect_nocrash "leading-dot '[ .5 .25 ]'"         '[ .5 .25 ]'
expect_nocrash "plus-sign '[ +1 +2 ]'"            '[ +1 +2 ]'
expect_nocrash "scientific '[ 1e3 2e-2 ]'"        '[ 1e3 2e-2 ]'
expect_nocrash "hex-ish '[ 0x10 ]'"               '[ 0x10 ]'
expect_nocrash "double-dot '[ 1.2.3 ]'"           '[ 1.2.3 ]'
expect_nocrash "literal inf '[ inf -inf ]'"       '[ inf -inf ]'
expect_nocrash "literal nan '[ nan ]'"            '[ nan ]'
expect_nocrash "empty literal '[ ]'"              '[ ]'

# =============================================================================
hr "2. Stack-underflow errors (spec R2: insufficient operands -> error)"
for op in + - '*' '<' '>' '=' '&' '|' m M '@' '.' c r f s o; do
  expect_err "binary '$op' on empty stack"        "$op"
done
for op in '!' R _ '#' S p d D '?'; do
  expect_err "unary '$op' on empty stack"         "$op"
done
expect_err "'+' with a single operand"            '[ 1 2 3 ] +'
expect_err "'-' with a single operand"            '[ 1 2 3 ] -'
expect_err "'@' with a single operand"            '[ 1 2 3 4 ] [ 2 2 ] r @'
expect_err "'\$' needs 3, given 2"                '[ 1 ] [ 1 ] $'
expect_err "'\$' needs 3, given 1"                '[ 1 ] $'
expect_err "'s' (swap) with 1 element"            '[ 1 ] s'
expect_err "'o' (over) with 1 element"            '[ 1 ] o'
expect_err "'r' (reshape) with only tensor"       '[ 1 2 3 4 ] r'
expect_err "'f' (fill) with only shape"           '[ 2 2 ] f'

# =============================================================================
hr "3. Type errors (spec R2: filename where tensor required & vice-versa)"
expect_err "string operand to '+'"                '[ 1 2 3 ] "x" +'
expect_err "two strings to '+'"                   '"a" "b" +'
expect_err "print a string"                       '"x" p'
expect_err "sum 'S' of a string"                  '"x" S'
expect_err "relu 'R' of a string"                 '"x" R'
expect_err "matmul with a string"                 '[ 1 2 3 4 ] [ 2 2 ] r "x" @'
expect_err "'(' (read pgm) given a tensor"        '[ 1 2 3 ] ('
expect_err "'{' (mmap load) given a tensor"       '[ 1 2 3 ] {'
expect_err "'?' (rand) given a string"            '"x" ?'
expect_err "reshape with string as shape"         '[ 1 2 3 4 ] "x" r'

# =============================================================================
hr "4. Shape-incompatibility errors (spec R2)"
expect_err "'+' different 1D sizes"               '[ 1 2 3 ] [ 1 2 ] +'
expect_err "'-' different 1D sizes"               '[ 1 2 ] [ 3 4 5 ] -'
expect_err "'*' different sizes"                  '[ 1 2 3 ] [ 1 2 ] *'
expect_err "'<' different sizes"                  '[ 1 2 3 ] [ 1 2 ] <'
expect_err "'&' different sizes"                  '[ 1 0 ] [ 1 0 1 ] &'
expect_err "'m' (min) different sizes"            '[ 1 2 ] [ 3 4 5 ] m'
expect_err "'M' (max) different sizes"            '[ 1 2 ] [ 3 4 5 ] M'
expect_err "'.' (dot) different-length vectors"   '[ 1 2 3 ] [ 1 2 ] .'
expect_err "'.' (dot) on 2D operands"             '[ 1 2 3 4 ] [ 2 2 ] r [ 1 2 3 4 ] [ 2 2 ] r .'
expect_err "'@' (matmul) on 1D operands"          '[ 1 2 3 ] [ 1 2 3 ] @'
expect_err "'@' incompatible inner dims (2x3 @ 2x3)" \
           '[ 1 2 3 4 5 6 ] [ 2 3 ] r [ 1 2 3 4 5 6 ] [ 2 3 ] r @'
expect_err "'c' (conv) on 1D operands"            '[ 1 2 3 ] [ 1 2 3 ] c'
expect_err "reshape element-count mismatch (3->4)" '[ 1 2 3 ] [ 2 2 ] r'
expect_err "reshape to >2 dims (MAX_DIM=2)"       '[ 1 2 3 4 5 6 ] [ 1 2 3 ] r'
expect_err "fill shape with >2 elements"          '[ 1 1 1 ] [ 1 ] f'
expect_err "rand shape with >2 elements"          '[ 1 1 1 ] ?'
expect_err "rand shape not 1D"                    '[ 1 2 3 4 ] [ 2 2 ] r ?'
expect_err "'\$' mask/operands size mismatch"     '[ 1 2 ] [ 3 4 ] [ 1 ] $'
# spec describes ')' for 2D tensors; feeding 1D is ambiguous -> only require survival
expect_nocrash "')' write-pgm on a 1D tensor"     '[ 1 2 3 ] "'"$TMP"'/bad.pgm" )'

# Overflow / huge-allocation guards (must error, must NOT crash/OOM-kill)
expect_err "reshape product overflow"             '[ 1 2 3 4 ] [ 100000 100000 ] r'
expect_err "fill enormous tensor"                 '[ 100000 100000 ] [ 1 ] f'
expect_err "rand enormous tensor"                 '[ 100000 100000 ] ?'
expect_nocrash "negative reshape dims"            '[ 1 2 3 4 ] [ -2 -2 ] r'

# I/O on bad paths (spec gives "file non apribile" as the canonical example)
expect_err "read pgm of missing file"             '"'"$TMP"'/nope.pgm" ('
expect_err "mmap-load missing file"               '"'"$TMP"'/nope.bin" {'

# =============================================================================
hr "5. Deterministic semantics (value oracles, format-robust)"
# operand order for non-commutative ops:  (-) is ( b a -- a-b ), a = top
expect_data "'-' order: [1 2 3] [4 5 6] - = [3 3 3]"  '[ 1 2 3 ] [ 4 5 6 ] - p' 'data=[3'
expect_data "'<' truth: 2<1 -> 0"                     '[ 1 ] [ 2 ] < p' 'data=[0'
expect_data "'<' truth: 1<2 -> 1"                     '[ 2 ] [ 1 ] < p' 'data=[1'
expect_data "'=' equal -> 1"                          '[ 5 ] [ 5 ] = p' 'data=[1'
expect_data "'!' negation: !1 -> 0"                   '[ 1 ] ! p' 'data=[0'
expect_data "'!' negation: !0 -> 1"                   '[ 0 ] ! p' 'data=[1'
expect_data "'&' AND 1&0 -> 0"                        '[ 1 ] [ 0 ] & p' 'data=[0'
expect_data "'|' OR 1|0 -> 1"                         '[ 1 ] [ 0 ] | p' 'data=[1'
expect_data "'\$' mask=1 selects a (=9)"              '[ 7 ] [ 9 ] [ 1 ] $ p' 'data=[9'
expect_data "'\$' mask=0 selects b (=7)"              '[ 7 ] [ 9 ] [ 0 ] $ p' 'data=[7'
expect_data "'m' min(3,5) -> 3"                       '[ 3 ] [ 5 ] m p' 'data=[3'
expect_data "'M' max(3,5) -> 5"                       '[ 3 ] [ 5 ] M p' 'data=[5'
expect_data "'R' relu([-2 3 -4]) sum -> 3"            '[ -2 3 -4 ] R S p' 'data=[3'
expect_data "'.' dot [1 2 3].[4 5 6] -> 32"           '[ 1 2 3 ] [ 4 5 6 ] . p' 'data=[32'
expect_data "'S' sum [1 2 3 4] -> 10"                 '[ 1 2 3 4 ] S p' 'data=[10'
expect_data "dup: [5 5] d + -> [10 10]"               '[ 5 5 ] d + p' 'data=[10'
expect_data "drop: keeps lower (=1)"                  '[ 1 ] [ 2 ] D p' 'data=[1'
expect_data "swap: [2] [8] s - -> -6"                 '[ 2 ] [ 8 ] s - p' 'data=[-6'
expect_data "over: [5] [9] o + + -> 19"               '[ 5 ] [ 9 ] o + + p' 'data=[19'
expect_data "fill repeat: [2 3][1 2] f sum -> 9"      '[ 2 3 ] [ 1 2 ] f S p' 'data=[9'
# matmul order/value:  a@b with a=[[1 2][3 4]] (top), b=[[0 1][0 0]] (lower)
#   a@b = [[0 1][0 3]] -> sum 4 ;  b@a (wrong order) = [[3 4][0 0]] -> sum 7
expect_data "matmul order+value (a@b sum == 4)" \
   '[ 0 1 0 0 ] [ 2 2 ] r [ 1 2 3 4 ] [ 2 2 ] r @ S p' 'data=[4'
# convolution with identity (delta) kernel returns input unchanged; diff sums 0
expect_data "conv(a, delta-kernel) == a  (diff sum 0)" \
   '[ 1 2 3 4 5 6 7 8 9 ] [ 3 3 ] r [ 0 0 0 0 1 0 0 0 0 ] [ 3 3 ] r c [ 1 2 3 4 5 6 7 8 9 ] [ 3 3 ] r - S p' \
   'data=[0'
# reshape memory-preserving roundtrip: reshape then ravel then re-shape, sum stable
expect_data "reshape/ravel preserve data (sum 21)" \
   '[ 1 2 3 4 5 6 ] [ 2 3 ] r _ S p' 'data=[21'
# multi-element results / shape op survive (display-only checks below)
expect_ok   "'#' shape of 2x3 prints"                 '[ 1 2 3 4 5 6 ] [ 2 3 ] r # p'
expect_ok   "newlines as token separators"            '[ 1 2 3 ]
[ 4 5 6 ]
+
p'

# randomness range: random < 2 elementwise must all be true -> sum == size
expect_data "rand values in range (sum of <2 == 100)" \
   '[ 100 ] [ 2 ] f [ 100 ] ? < S p' 'data=[100'

# =============================================================================
hr "6. On-disk tensor format (spec R4: data aligned at offset 64)"
BINF="$TMP/disk.bin"
expect_ok "save 2x3 tensor via '}'" '[ 1 2 3 4 5 6 ] [ 2 3 ] r "'"$BINF"'" }'
if [ -f "$BINF" ]; then
  # ndim (int32 at struct offset 8) must be 2
  nd="$(od -An -tu4 -N4 -j8 "$BINF" 2>/dev/null | tr -d ' \n')"
  [ "$nd" = "2" ] && note_pass "disk: ndim==2 at offset 8" \
                  || note_fail "disk: ndim expected 2, got '$nd'"
  # data_offset field == 64  (off_t, 8-byte aligned -> struct offset 16;
  # accept offset 12 too in case off_t/packing differs)
  d16="$(od -An -tu8 -N8 -j16 "$BINF" 2>/dev/null | tr -d ' \n')"
  d12="$(od -An -tu4 -N4 -j12 "$BINF" 2>/dev/null | tr -d ' \n')"
  if [ "$d16" = "64" ] || [ "$d12" = "64" ]; then
    note_pass "disk: data_offset field == 64"
  else
    note_fail "disk: data_offset field != 64 (got @16='$d16' @12='$d12')"
  fi
  # the float data must physically begin at byte 64: first float == 1.0f
  first="$(od -An -tx1 -N4 -j64 "$BINF" 2>/dev/null | tr -d ' \n')"
  [ "$first" = "0000803f" ] && note_pass "disk: data@64 == 1.0f (00 00 80 3f)" \
                            || note_fail "disk: bytes@64 expected 0000803f, got '$first'"
  # total size == 64 header/padding + 6 floats * 4 bytes == 88
  sz=$(stat -c%s "$BINF" 2>/dev/null)
  [ "$sz" = "88" ] && note_pass "disk: file size == 88" \
                   || note_fail "disk: size expected 88, got '$sz'"
fi
# mmap roundtrip: load it back and verify content survives (sum 1..6 == 21)
expect_data "mmap '{' roundtrip sum == 21" '"'"$BINF"'" { S p' 'data=[21'

# pgm roundtrip (8-bit quantized; only structural / no-crash guarantees)
PGMF="$TMP/img.pgm"
expect_ok "write pgm via ')'" '[ 4 4 ] [ 0.5 ] f "'"$PGMF"'" )'
expect_ok "read pgm via '('"  '"'"$PGMF"'" ( p'

# =============================================================================
hr "7. Provided example programs (spec: must run correctly)"
if [ -d "$ROOT/examples" ]; then
  for ex in "$ROOT"/examples/*.tensorforth; do
    [ -e "$ex" ] || continue
    name="$(basename "$ex")"
    OUT="$(cd "$ROOT" && timeout 60 "$BIN" "$ex" 2>&1)"; RC=$?
    if   [ "$RC" -ge 128 ]; then note_fail "example CRASH rc=$RC :: $name"
    elif [ "$RC" -eq 0 ];   then note_pass "example runs: $name"
    else note_fail "example failed rc=$RC :: $name :: ${OUT:0:100}"; fi
  done
fi

# =============================================================================
hr "9. Spec-conformance findings (expected to expose defects)"
# Spec: "I token sono separati da spazi o da ritorni a capo." Plural 'spazi'
# => multiple consecutive spaces are still a valid separator.
expect_ok   "multiple spaces between tokens are a valid separator" \
            '[ 1 2 3 ]  [ 4 5 6 ] + p'
expect_ok   "double space inside a literal is a valid separator" \
            '[ 1  2 3 ] p'
# Spec: a 1D tensor's shape is [n] (see '#'); 'p' must report the same shape.
run_prog '[ 1 2 3 ] p'
pn="$(printf '%s' "$OUT" | tr -d ' \t')"
if printf '%s' "$pn" | grep -qF 'shape=[3]'; then
  note_pass "'p' reports 1D shape as [3] (consistent with '#')"
else
  note_fail "'p' / '#' shape inconsistency: p says $(printf '%s' "$OUT" | grep -o 'shape=\[[^]]*\]'), # says [3]"
fi
# Spec: '@' requires both operands 2D & dim-compatible. A 1xN built by reshape
# is 2D, so [1x3] @ [3x1] -> [1x1] is well-defined and must be accepted.
expect_ok   "'@' accepts valid 2D matrices with a unit dimension (1x3 @ 3x1)" \
            '[ 1 1 1 ] [ 3 1 ] r [ 1 2 3 ] [ 1 3 ] r @ p'

# =============================================================================
hr "8. Resource release (spec R5) -- valgrind, if available"
if command -v valgrind >/dev/null 2>&1; then
  cat > "$TMP/stress.tf" <<'EOF'
[ 60 60 ] ? [ 60 60 ] ? @ d S p
[ 8 8 ] ? [ 3 3 ] [ 0.11 ] f c S p
[ 1 2 3 4 5 6 ] [ 2 3 ] r _ # p
EOF
  vout="$(cd "$ROOT" && OMP_NUM_THREADS=1 valgrind --quiet \
            --error-exitcode=99 --leak-check=full \
            --errors-for-leak-kinds=definite,indirect \
            "$BIN" "$TMP/stress.tf" 2>&1)"; vrc=$?
  defl="$(printf '%s' "$vout" | grep -oE 'definitely lost: [0-9,]+ bytes' | head -1)"
  indl="$(printf '%s' "$vout" | grep -oE 'indirectly lost: [0-9,]+ bytes' | head -1)"
  if [ "$vrc" -ne 99 ] \
     && { [ -z "$defl" ] || printf '%s' "$defl" | grep -q ': 0 bytes'; } \
     && { [ -z "$indl" ] || printf '%s' "$indl" | grep -q ': 0 bytes'; }; then
    note_pass "valgrind: no definite/indirect leaks, no memory errors"
  else
    note_fail "valgrind issue (rc=$vrc) :: ${defl:-?} / ${indl:-?} :: $(printf '%s' "$vout" | grep -E 'ERROR SUMMARY|Invalid' | head -2)"
  fi
else
  printf '  %s valgrind not installed -- skipped\n' "$(y SKIP)"
fi

# =============================================================================
printf '\n\033[1m=============== SUMMARY ===============\033[0m\n'
printf 'passed: %s   failed: %s\n' "$(g "$PASS")" "$([ "$FAIL" -eq 0 ] && g 0 || r "$FAIL")"
if [ "$FAIL" -gt 0 ]; then
  printf '\n\033[1mFailures:\033[0m\n'
  for f in "${FAILED[@]}"; do printf '  - %s\n' "$f"; done
  exit 1
fi
printf '\n%s\n' "$(g 'ALL HARD-BLACK-BOX CHECKS PASSED')"
