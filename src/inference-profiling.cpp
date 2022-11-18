// Implementation for type inference profiling

#include "julia.h"
#include "julia_internal.h"

#include <vector>

using std::vector;


jl_mutex_t jl_typeinf_profiling_lock;

// Guarded by jl_typeinf_profiling_lock.
vector<jl_value_t*> inference_profiling_results;

// == exported interface ==

extern "C" {

JL_DLLEXPORT jl_array_t* jl_typeinf_profiling_clear_and_fetch()
{
    JL_LOCK(&jl_typeinf_profiling_lock);

    size_t len = inference_profiling_results.size();

    jl_array_t *out = jl_alloc_array_1d(jl_array_any_type, len);
    memcpy(out->data, inference_profiling_results.data(), len * sizeof(void*));

    inference_profiling_results.clear();

    JL_UNLOCK(&jl_typeinf_profiling_lock);

    return out;
}

JL_DLLEXPORT void jl_typeinf_profiling_push_timing(jl_value_t *timing)
{
    JL_LOCK(&jl_typeinf_profiling_lock);

    inference_profiling_results.push_back(timing);

    JL_UNLOCK(&jl_typeinf_profiling_lock);
}

}  // extern "C"
