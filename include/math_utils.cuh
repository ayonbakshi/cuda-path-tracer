#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#define EPSILON 1e-10f
#define INF 1e10f
#define M_PI_F 3.14159265358979323846f

class vec3 {
public:
    float p[3];

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float t) { p[0]=p[1]=p[2]=t; }
    __host__ __device__ vec3(float x, float y, float z) { p[0] = x; p[1] = y; p[2] = z; }
    __host__ __device__ inline float x() const { return p[0]; }
    __host__ __device__ inline float y() const { return p[1]; }
    __host__ __device__ inline float z() const { return p[2]; }
    __host__ __device__ inline float r() const { return p[0]; }
    __host__ __device__ inline float g() const { return p[1]; }
    __host__ __device__ inline float b() const { return p[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-p[0], -p[1], -p[2]); }

    __host__ __device__ inline float operator[](int i) const { return p[i]; }
    __host__ __device__ inline float& operator[](int i) { return p[i]; };

    __host__ __device__ inline float norm() const { return sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]); }
    __host__ __device__ inline float sqrNorm() const { return p[0]*p[0] + p[1]*p[1] + p[2]*p[2]; }
    __host__ __device__ inline vec3 normalize() {
        float k = 1.0 / sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        p[0] *= k; p[1] *= k; p[2] *= k;
        return *this;
    }
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.p[0] >> t.p[1] >> t.p[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.p[0] << " " << t.p[1] << " " << t.p[2];
    return os;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.p[0] + v2.p[0], v1.p[1] + v2.p[1], v1.p[2] + v2.p[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.p[0] - v2.p[0], v1.p[1] - v2.p[1], v1.p[2] - v2.p[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.p[0] * v2.p[0], v1.p[1] * v2.p[1], v1.p[2] * v2.p[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.p[0] / v2.p[0], v1.p[1] / v2.p[1], v1.p[2] / v2.p[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.p[0], t*v.p[1], t*v.p[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.p[0]/t, v.p[1]/t, v.p[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.p[0], t*v.p[1], t*v.p[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.p[0] *v2.p[0] + v1.p[1] *v2.p[1]  + v1.p[2] *v2.p[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1.p[1]*v2.p[2] - v1.p[2]*v2.p[1]),
                (-(v1.p[0]*v2.p[2] - v1.p[2]*v2.p[0])),
                (v1.p[0]*v2.p[1] - v1.p[1]*v2.p[0]));
}

template <typename T>
__host__ __device__ inline void swap(T &a, T &b){
    T c(a); a=b; b=c;
}

#endif