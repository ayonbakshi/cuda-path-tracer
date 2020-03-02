#include "../include/camera.cuh"
#include "../include/math_utils.cuh"

__host__ __device__
Camera::Camera(int width, int height, float fov, const vec3 &origin, const vec3 &dir)
    : width{width}, height{height}, fov{fov * M_PI_F / 180.0f},
    origin{origin}, dir{dir}
    {
        inv_width = 1.0f / width;
        inv_height = 1.0f / height;
        aspect_ratio = width / float(height);
        angle = tan(0.5f * fov);

        calc_axes();
    }

__host__ __device__
void Camera::move(const vec3 &new_origin, const vec3 &new_dir) {
    origin = new_origin;
    dir = new_dir;
    dir.normalize();
    calc_axes();
}

__host__ __device__
void Camera::move_from_to(const vec3 &from, const vec3 &to) {
    move(from, to - from);
}

__host__ __device__
vec3 Camera::ray_dir_at_pixel(float x, float y) const {
    float xx = (2 * x * inv_width - 1) * angle * aspect_ratio; 
    float yy = (1 - 2 * y * inv_height) * angle;

    vec3 img_pt = right * xx + up * yy + dir;
    img_pt.normalize();
    return img_pt;
}

__host__ __device__
void Camera::calc_axes() {
    // approximately "up"
    vec3 tmp_up = {0, 1, 0};
    tmp_up.normalize();
    right = cross(dir, tmp_up);
    right.normalize();
    up = cross(right, dir);
}