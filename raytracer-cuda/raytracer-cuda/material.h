#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>
#include "ray.h"
#include "hitable.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
	vec3 p;
	do {
		/// reduction alogrithm
		p = 2.0f * RANDVEC3 - vec3(1, 1, 1); // [-1, 1] x [-1, 1] x [-1, 1] unit cube
	} while (p.squared_length() >= 1.0f);
	return p;
}

class material {
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertian : public material {
public:
	__device__ lambertian(const vec3& a) : albedo(a) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		scattered = ray(rec.p, rec.normal + random_in_unit_sphere(local_rand_state)); // world coordinates
		attenuation = albedo;
		return true;
	}
private:
	vec3 albedo;
};

class metal : public material {
public:
	__device__ metal(const vec3& a, const float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1.0f; }
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		vec3 R, V, N;
		
		N = rec.normal;
		V = unit_vector(r_in.direction());
		R = V - 2.0f * dot(V, N) * N;
		scattered = ray(rec.p, R + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), N) > 0.0f);
	}
private:
	vec3 albedo;
	float fuzz;
};

#endif // !MATERIAL_H
