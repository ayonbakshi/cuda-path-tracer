#include "../include/object.cuh"
#include "../include/math_utils.cuh"


__host__ __device__
Object::Object(const Material &mat): mat{mat} {}

__host__ __device__
Sphere::Sphere(const vec3 &center, float radius, const Material &mat):
    Object{mat}, center{center}, radius{radius} {}

__host__ __device__
bool Sphere::intersect(const vec3 &ray_orig,
                          const vec3 &ray_dir,
                          float &dist,
                          vec3 &hit_loc,
                          vec3 &hit_norm) const
{
    float t0, t1;
    vec3 L = center - ray_orig;
    float tca = dot(L, ray_dir); 
    // if (tca < 0) return false;
    float d2 = L.sqrNorm() - tca * tca; 
    if (d2 > radius * radius) return false; 
    float thc = sqrt(radius * radius - d2); 
    t0 = tca - thc; 
    t1 = tca + thc; 

    if (t1 < t0) swap(t0, t1);
    if (t1 < EPSILON) return false;
    
    dist = t0 < EPSILON ? t1 : t0;

    hit_loc = ray_orig + (ray_dir * dist);
    hit_norm = hit_loc - center;
    return true; 
}

__host__ __device__
Plane::Plane(const vec3 &normal, const vec3 &center, float size, const Material &mat):
    Object{mat}, normal{normal}, center{center}, size{size}
{
    this->normal.normalize();
}

__host__ __device__
bool Plane::intersect(const vec3 &ray_orig,
                          const vec3 &ray_dir,
                          float &dist,
                          vec3 &hit_loc,
                          vec3 &hit_norm) const
{
    if(abs(dot(ray_dir, normal)) < EPSILON) return false;
        
    dist = dot((center - ray_orig), normal) / dot(ray_dir, normal);
    if(dist < EPSILON) return false;

    hit_loc = ray_orig + (ray_dir * dist);

    vec3 to_center = center - hit_loc;
    if(size != INF && to_center.sqrNorm() > size * size) return false;

    hit_norm = normal;
    return true;
}