#ifndef CAMERA_H
#define CAMERA_H

#include "math_utils.cuh"

class Camera {
    int width, height;
    float inv_width, inv_height;
    float aspect_ratio;
    float fov, angle;
    vec3 origin, dir;
    vec3 up, right;

public:
    __host__ __device__ Camera(int width, int height, float fov, const vec3 &origin, const vec3 &dir);
    __host__ __device__ void move(const vec3 &new_origin, const vec3 &new_dir);
    __host__ __device__ void move_from_to(const vec3 &from, const vec3 &to);
    __host__ __device__ vec3 ray_dir_at_pixel(float x, float y) const;
    __host__ __device__ const vec3 &get_origin() const { return origin; }
    __host__ __device__ int get_width() const { return width; }
    __host__ __device__ int get_height() const { return height; }
private:
    __host__ __device__ void calc_axes();
};

#endif