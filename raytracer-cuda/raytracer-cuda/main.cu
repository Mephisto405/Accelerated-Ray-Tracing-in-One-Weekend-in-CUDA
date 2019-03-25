#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define SPP 100
#define MAX_DEPTH 64

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 renderPixel(const ray& r, hitable_list **d_world, curandState *rand_state) {
	vec3 throughput(1, 1, 1);
	vec3 bsdfWeight;
	ray in = r;
	ray out;
	int depth = 0;

	while (depth <= MAX_DEPTH || MAX_DEPTH < 0) {
		hit_record rec;

		if (d_world[0]->hit(in, 0.001f, FLT_MAX, rec)) {
			if (rec.mat_ptr->scatter(in, rec, bsdfWeight, out, rand_state)) {
				throughput *= bsdfWeight;
				in = out;
			}
			else {
				return vec3(0, 0, 0);
			}
		}
		else {
			vec3 u = unit_vector(in.direction());
			float w = 0.5f * (u.y() + 1.0f);
			vec3 value = (1.0f - w) * vec3(1.0, 1.0, 1.0) + w * vec3(0.5, 0.7, 1.0);
			return throughput * value;
		}

		depth++;
	}

	return vec3(0, 0, 0); 
}

__global__ void render(vec3 *fb, int max_x, int max_y,
						camera **cam, 
						hitable_list **d_world,
						curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];

	vec3 spec(0, 0, 0);
	for (int s = 0; s < SPP; s++) {
		/// normalized uv coordinate
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v);
		spec += renderPixel(r, d_world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;

	/// gamma correction
	spec /= float(SPP);
	spec[0] = sqrt(spec[0]);
	spec[1] = sqrt(spec[1]);
	spec[2] = sqrt(spec[2]);
	fb[pixel_index] = spec;
}

__global__ void create_world(hitable **d_list, hitable_list **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
			new lambertian(vec3(0.8, 0.3, 0.3)));
		d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
			new lambertian(vec3(0.8, 0.8, 0.0)));
		d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
			new metal(vec3(0.8, 0.6, 0.2), 1.0));
		d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
			new metal(vec3(0.8, 0.8, 0.8), 0.3));
		*d_world = new hitable_list(d_list, 4);
		*d_camera = new camera();
	}
}

__global__ void free_world(hitable **d_list, hitable_list **d_world, camera **d_camera) {
	delete d_list[0];
	delete d_list[1];
	delete d_world[0];
	delete d_camera[0];
}

__global__ void camere_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	int pixel_index = j * max_x + i;
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

int main() {
	/*
	Initiliaztion
	*/
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;
	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);
	hitable **d_list; // list of hitable pointer
	hitable_list **d_world;
	curandState *d_rand_state;
	camera **d_camera;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	// File open
	ofstream file;
	file.open("image.ppm");

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	/*
	Scene Initializer
	*/
	checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable_list *)));
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world << <1, 1 >> >(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/*
	Camera Initiailizer
	*/
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	camere_init << <blocks, threads >> >(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/*
	Setting thread blocks and Run
	*/
	clock_t start, stop;
	start = clock();

	render<<<blocks, threads>>>(fb, nx, ny, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	/*
	Output FB as Image
	*/
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			vec3 col = fb[pixel_index];
			int ir = int(255.99f*col[0]);
			int ig = int(255.99f*col[1]);
			int ib = int(255.99f*col[2]);
			file << ir << " " << ig << " " << ib << "\n";
		}
	}

	file.close();

	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> >(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}