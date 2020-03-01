#ifndef SCENE_H
#define SCENE_H

#include "object.cuh"
#include "math_utils.cuh"

class Scene {
public:
    int num_objects;
    Object **objects;
    
    __host__ __device__
    Scene(int num_objects, Object **objects): num_objects{num_objects}, objects{objects} {}

    __device__
    const Object *hit_scene(const vec3 &ray_orig,
                            const vec3 &ray_dir,
                            vec3 &hit_loc,
                            vec3 &hit_norm)
    {
        float min_dist = INF;
        const Object *closest_obj = 0;

        // calculate closest obj
        for(int i = 0; i < num_objects; ++i){
            float dist = INF;
            vec3 tmp_hit_loc, tmp_hit_norm;
            if(objects[i]->intersect(ray_orig, ray_dir, dist, tmp_hit_loc, tmp_hit_norm)){
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_obj = objects[i];
                    hit_loc = tmp_hit_loc;
                    hit_norm = tmp_hit_norm;
                }
            }
        }
        return closest_obj;
    }

};

#endif