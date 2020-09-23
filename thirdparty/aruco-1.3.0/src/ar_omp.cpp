#ifndef USE_OMP
int32_t omp_get_max_threads() { return 1; }
int32_t omp_get_thread_num() { return 0; }
#endif
