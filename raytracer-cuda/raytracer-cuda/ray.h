#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
	__device__ ray() {}
	__device__ ray(const vec3& point, const vec3& direction) { o = point; d = direction; }
	__device__ vec3 origin() const { return o; }
	__device__ vec3 direction() const { return d; }
	__device__ vec3 point_at_t(float t) const { return o + t*d; }
private:
	/* origin of this ray */
	vec3 o;
	/* direction of this ray */
	vec3 d;
};

#endif