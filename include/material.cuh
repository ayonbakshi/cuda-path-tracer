#ifndef MATERIAL_H
#define MATERIAL_H

#include "math_utils.cuh"

class Material {
    vec3 color;
public:
    __host__ __device__ Material(const vec3 &color): color{color} {}
    
    __host__ __device__ vec3 get_col() const { return color; }
    
};

#endif