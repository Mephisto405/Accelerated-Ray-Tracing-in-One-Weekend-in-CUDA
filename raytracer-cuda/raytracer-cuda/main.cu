#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "hitable_list.h"
#include "float.h"
#include "sphere.h"

using namespace std;

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

__device__ vec3 renderPixel(const ray& r, hitable_list **d_world) {
	float w, tmin, tmax;
	vec3 u, N;
	hit_record rec;
	bool hit;

	tmin = 0.0f;
	tmax = FLT_MAX;

	hit = d_world[0]->hit(r, tmin, tmax, rec);

	if (hit) {
		return 0.5f * (rec.normal + vec3(1, 1, 1));
	}
	else {
		u = unit_vector(r.direction());
		w = 0.5f * (u.y() + 1.0f);
		return (1.0f - w) * vec3(1.0, 1.0, 1.0) + w * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void render(vec3 *fb, int max_x, int max_y,
						vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, 
						hitable_list **d_world) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;

	/// normalized uv coordinate
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);

	ray r(origin, lower_left_corner + u*horizontal + v*vertical);
	fb[pixel_index] = renderPixel(r, d_world);
}

__global__ void create_world(hitable **d_list, hitable_list **d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5); // d_list[0]
		*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
		*(d_world) = new hitable_list(d_list, 2);
	}
}

__global__ void free_world(hitable **d_list, hitable_list **d_world) {
	delete d_list[0];
	delete d_list[1];
	delete d_world[0];
}

int main() {
	/*
	Initiliaztion
	*/
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;
	vec3 lower_left_corner(-2.0, -1.0, -1.0);
	vec3 horizontal(4.0, 0.0, 0.0);
	vec3 vertical(0.0, 2.0, 0.0);
	vec3 origin(0.0, 0.0, 0.0);
	int num_pixels = nx*ny;
	size_t fb_size = num_pixels*sizeof(vec3);
	hitable **d_list; // list of hitable pointer
	hitable_list **d_world;

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
	create_world<<<1, 1>>>(d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	/*
	Setting thread blocks and Run
	*/
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render<<<blocks, threads>>>(fb, nx, ny, 
		origin, lower_left_corner, horizontal, vertical,
		d_world);
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
	free_world << <1, 1 >> >(d_list, d_world);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}