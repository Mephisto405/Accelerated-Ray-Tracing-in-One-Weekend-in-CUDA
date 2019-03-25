#ifndef SPHERE_H
#define SPHERE_H

#include "hitable.h"
#include "ray.h"
#include "vec3.h"
#include "material.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 _center, float _radius, material *m) : center(_center), radius(_radius), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
private:
	vec3 center;
	float radius;
	material *mat_ptr;
};

__device__
bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - a*c;

	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < tmax && temp > tmin) {
			rec.t = temp;
			rec.p = r.point_at_t(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < tmax && temp > tmin) {
			rec.t = temp;
			rec.p = r.point_at_t(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

#endif