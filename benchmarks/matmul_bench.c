#define _POSIX_C_SOURCE 200809L
#include "parser.h"
#include <cblas.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(EXIT_FAILURE);
}

static int argument(const char *s, int low, int high) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v < low || v > high)
        fail("invalid argument (N: 2..20000, repeats: 1..99, threads: 1..64)");
    return (int)v;
}

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock_gettime");
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float random_value(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return (float)(*state >> 8) * (1.0f / 16777216.0f);
}

static array_instance *tensor(int n, uint32_t *state) {
    size_t count = (size_t)n * (size_t)n;
    float *data = malloc(count * sizeof(*data));
    if (!data) fail("input allocation");
    for (size_t i = 0; i < count; ++i) data[i] = random_value(state);
    shape_t shape = {n, n, 2};
    array_instance *result = new_instance(data, shape);
    if (!result) fail("tensor allocation");
    return result;
}

static array_instance *run_tf(stack *s, array_instance *a,
                              array_instance *b, double *elapsed) {
    // @ computes TOS @ second: push B then A to obtain the same A*B as BLAS.
    // Owner references retain the inputs across calls, outside the timer.
    if (stack_push_instance(s, b) || stack_push_instance(s, a))
        fail("stack preparation");
    double start = now();
    int status = parser("@", s);
    *elapsed = now() - start;
    if (status != TF_OK) fail("TensorForth @");
    array_instance *result = stack_pop(s);
    if (!result || s->top != 0) fail("unexpected runtime stack");
    return result;
}

static void run_blas(int n, const float *a, const float *b, float *c,
                     double *elapsed) {
    double start = now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
    *elapsed = now() - start;
}

static double validate(int n, const float *a, const float *b,
                       const float *tf, const float *blas) {
    double max_relative = 0.0;
    size_t count = (size_t)n * (size_t)n;
    for (size_t i = 0; i < count; ++i) {
        double error = fabs((double)tf[i] - (double)blas[i]);
        if (!isfinite(tf[i]) || !isfinite(blas[i]) ||
            error > 1e-4 + 1e-4 * fabs((double)blas[i]))
            fail("full TensorForth/OpenBLAS output comparison failed");
        double relative = error / fmax(fabs((double)blas[i]), 1e-12);
        if (relative > max_relative) max_relative = relative;
    }
    // Independent double-precision dot products also detect operand reversal.
    for (size_t sample = 0; sample < 16; ++sample) {
        size_t row = (sample * 313 + 1) % (size_t)n;
        size_t col = (sample * 719 + 2) % (size_t)n;
        double reference = 0.0;
        for (size_t k = 0; k < (size_t)n; ++k)
            reference += (double)a[row * (size_t)n + k] *
                         (double)b[k * (size_t)n + col];
        size_t index = row * (size_t)n + col;
        double tolerance = 1e-4 + 1e-4 * fabs(reference);
        if (fabs((double)tf[index] - reference) > tolerance ||
            fabs((double)blas[index] - reference) > tolerance)
            fail("independent double-precision reference failed");
    }
    return max_relative;
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void summary(const char *name, double *times, int count, int n) {
    qsort(times, (size_t)count, sizeof(*times), compare);
    double median = count % 2 ? times[count / 2] :
        (times[count / 2 - 1] + times[count / 2]) / 2.0;
    double work = 2.0 * (double)n * (double)n * (double)n;
    printf("summary,%s,median_s,%.9f,GFLOPS,%.6f,min_s,%.9f,max_s,%.9f\n",
           name, median, work / median / 1e9, times[0], times[count - 1]);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s N repeats threads\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = argument(argv[1], 2, 20000);
    int repeats = argument(argv[2], 1, 99);
    int threads = argument(argv[3], 1, 64);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    openblas_set_num_threads(threads);
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("N,%d,repeats,%d,warmups,1,requested_threads,%d,seed,123456789\n",
           n, repeats, threads);
    printf("OpenBLAS,%s,core,%s,threads,%d\n", openblas_get_config(),
           openblas_get_corename(), openblas_get_num_threads());
    int team = 0;
    #pragma omp parallel
    {
        #pragma omp single
        team = omp_get_num_threads();
    }
    printf("OpenMP_team,%d\n", team);
    if (team != threads || openblas_get_num_threads() != threads)
        fail("thread count mismatch");

    uint32_t state = 123456789;
    array_instance *a = tensor(n, &state), *b = tensor(n, &state);
    size_t count = (size_t)n * (size_t)n;
    float *c = malloc(count * sizeof(*c));
    if (!c) fail("OpenBLAS output allocation");
    // Pre-touch BLAS output outside the timed region, as for a prepared buffer.
    memset(c, 0, count * sizeof(*c));
    stack *s = stack_init();
    if (!s) fail("stack allocation");
    double tf_times[99], blas_times[99];
    double worst = 0.0;
    puts("phase,round,TensorForth_s,OpenBLAS_s,max_relative_error");
    for (int round = -1; round < repeats; ++round) {
        double tf_elapsed, blas_elapsed;
        array_instance *result;
        // Alternate order to reduce systematic thermal/order bias.
        if (round % 2 == 0) {
            run_blas(n, a->data, b->data, c, &blas_elapsed);
            result = run_tf(s, a, b, &tf_elapsed);
        } else {
            result = run_tf(s, a, b, &tf_elapsed);
            run_blas(n, a->data, b->data, c, &blas_elapsed);
        }
        double error = validate(n, a->data, b->data, result->data, c);
        if (error > worst) worst = error;
        printf("%s,%d,%.9f,%.9f,%.9g\n", round < 0 ? "warmup" : "sample",
               round + 1, tf_elapsed, blas_elapsed, error);
        instance_free(result); // Excluded, as is the language's D opcode.
        if (round >= 0) {
            tf_times[round] = tf_elapsed;
            blas_times[round] = blas_elapsed;
        }
    }
    summary("TensorForth", tf_times, repeats, n);
    summary("OpenBLAS", blas_times, repeats, n);
    printf("validation,PASS,full_output_each_round,reference_samples,16,max_relative_error,%.9g\n", worst);
    stack_free(s);
    instance_free(a);
    instance_free(b);
    free(c);
    return EXIT_SUCCESS;
}
