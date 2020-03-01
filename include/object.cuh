#ifndef OBJECT_H
#define OBJECT_H

#include "material.cuh"

class Object {
    public:
        Material mat; // replace with full mat after
        __host__ __device__ Object(const Material &mat);
        __host__ __device__ virtual ~Object() {}

        __host__ __device__
        virtual bool intersect(const vec3 &orig, const vec3 &dir,
                               float &dist,
                               vec3& hit_loc, vec3 &hit_norm) const = 0;
};

class Sphere : public Object {
    vec3 center;
    float radius;

public:
    __host__ __device__ Sphere(const vec3 &center, float radius, const Material &mat);
    __host__ __device__ ~Sphere() {}

    __host__ __device__
    const vec3 &get_center() const { return center; }
    __host__ __device__
    const float &get_radius() const { return radius; }
    
    __host__ __device__
    bool intersect(const vec3 &orig, const vec3 &dir,
                   float &dist,
                   vec3& hit_loc, vec3 &hit_norm) const;
};

class Plane : public Object {
    vec3 normal, center;
    float size;

public:
    __host__ __device__ Plane(const vec3 &center, const vec3 &normal, float size, const Material &mat);
    __host__ __device__ ~Plane() {}

    __host__ __device__
    bool intersect(const vec3 &orig, const vec3 &dir,
                   float &dist,
                   vec3& hit_loc, vec3 &hit_norm) const;
};

#endif