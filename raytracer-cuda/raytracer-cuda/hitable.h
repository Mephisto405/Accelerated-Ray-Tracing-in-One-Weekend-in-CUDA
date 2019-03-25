#ifndef HITABLE_H
#define HITABLE_H

class material;

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class hitable {
public:
	///  You can make a function pure virtual with =0 at the end of function declaration. 
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif