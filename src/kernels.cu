#include <iostream>
#include <time.h>

#include "../include/math_utils.cuh"
#include "../include/object.cuh"
#include "../include/material.cuh"
#include "../include/scene.cuh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ inline vec3 trace(const vec3 &orig, const vec3 &dir, Scene *scene) {
    vec3 hit_loc, hit_norm;
    const Object *closest = scene->hit_scene(orig, dir, hit_loc, hit_norm);
    // const Object *closest = 0;
    if (closest) {
        return hit_norm;
    } else {
        return vec3(0.1,0.3,0.8);
    }
}

__global__ void render(vec3 *fb, Scene **scene, /*camera stuff*/ int max_x, int max_y, float invWidth, float invHeight, float angle, float aspectratio) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    float xx = (2 * ((i + 0.5) * invWidth) - 1) * angle * aspectratio; 
    float yy = (1 - 2 * ((j + 0.5) * invHeight)) * angle; 
    vec3 raydir(xx, yy, -1); 
    raydir.normalize();

    vec3 orig(0);

    fb[j*max_x + i] = trace(orig, raydir, *scene);
}

__global__
void create_scene(Scene **scene, Object **d_list, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Material m(vec3(0.5));
        *(d_list)   = new Sphere(vec3(0,0,-5), 1, m);

        *scene = new Scene(num_objects, d_list);
    }
}

void run(int width, int height, int tx, int ty, float *pixels) {
    int num_pixels = width*height;
    size_t fb_size = num_pixels*sizeof(vec3);



    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Render our buffer
    dim3 blocks(width/tx+1,height/ty+1);
    dim3 threads(tx,ty);

    int num_objects = 1;
    Object **objects;
    checkCudaErrors(cudaMalloc((void **)&objects, num_objects*sizeof(Object*)));

    Scene **scene;
    checkCudaErrors(cudaMalloc((void **)&scene, sizeof(Scene*)));

    create_scene<<<1,1>>>(scene, objects, num_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // replace this with camera stuff later
    float invWidth = 1 / float(width), invHeight = 1 / float(height); 
    float fov = 30, aspectratio = width / float(height); 
    float angle = tan(M_PI * 0.5 * fov / 180.); 


    render<<<blocks, threads>>>(fb, scene, width, height, invWidth, invHeight, angle, aspectratio);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // converts vec3 to 3 floats since vec3 is POD
    checkCudaErrors(cudaMemcpy(pixels, fb, fb_size, cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(fb));
}