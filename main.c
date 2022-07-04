#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#undef complex
#include "minfft.h"


typedef enum 
{
    LDR,
    HDR
} format_t;

typedef struct
{
    format_t format;
    int width, height, channels;
    unsigned char *ldr_buffer;
    float *hdr_buffer;
} image_t;

typedef struct {
    int width, height;
    unsigned char *color_buffer;
    float *depth_buffer;
} framebuffer_t;


typedef struct window_t
{
    HWND handle;
    HDC memory_dc;
    image_t *surface;
	
    int should_close;
    char keys[256];
    void *userdata;
} window_t;

/* platform initialization */

static int g_initialized = 0;

#ifdef UNICODE
static const wchar_t *const WINDOW_CLASS_NAME = L"Class";
static const wchar_t *const WINDOW_ENTRY_NAME = L"Entry";
#else
static const char *const WINDOW_CLASS_NAME = "Class";
static const char *const WINDOW_ENTRY_NAME = "Entry";
#endif

#define EPSILON 1e-5f
#define PI 3.1415927f

#define TO_RADIANS(degrees) ((PI / 180) * (degrees))
#define TO_DEGREES(radians) ((180 / PI) * (radians))

#define LINE_SIZE 256
#define PATH_SIZE 256

#define UNUSED_VAR(x) ((void)(x))
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))



typedef struct vec2_t
{
    float x, y;
} vec2_t;

typedef struct vec3_t
{
    float x, y, z;
} vec3_t;

typedef struct vec4_t
{
    float x, y, z, w;
} vec4_t;

typedef struct quat_t 
{
    float x, y, z, w;
} quat_t;

typedef struct mat3_t 
{
    float m[3][3];
} mat3_t;

typedef struct mat4_t
{
    float m[4][4];
} mat4_t;


/* float related functions */

float float_min(float a, float b) {
    return a < b ? a : b;
}

float float_max(float a, float b) {
    return a > b ? a : b;
}

float float_lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float float_clamp(float f, float min, float max) {
    return f < min ? min : (f > max ? max : f);
}

float float_saturate(float f) {
    return f < 0 ? 0 : (f > 1 ? 1 : f);
}

float float_from_uchar(unsigned char value) {
    return value / 255.0f;
}

unsigned char float_to_uchar(float value) {
    return (unsigned char)(value * 255);
}

float float_srgb2linear(float value) {
    return (float)pow(value, 2.2);
}

float float_linear2srgb(float value) {
    return (float)pow(value, 1 / 2.2);
}

/*
 * for aces filmic tone mapping curve, see
 * https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
 */
float float_aces(float value) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    value = (value * (a * value + b)) / (value * (c * value + d) + e);
    return float_saturate(value);
}

void float_print(const char *name, float f) {
    printf("float %s = %f\n", name, f);
}

/* vec2 related functions */

vec2_t vec2_new(float x, float y) {
    vec2_t v;
    v.x = x;
    v.y = y;
    return v;
}

vec2_t vec2_min(vec2_t a, vec2_t b) {
    float x = float_min(a.x, b.x);
    float y = float_min(a.y, b.y);
    return vec2_new(x, y);
}

vec2_t vec2_max(vec2_t a, vec2_t b) {
    float x = float_max(a.x, b.x);
    float y = float_max(a.y, b.y);
    return vec2_new(x, y);
}

vec2_t vec2_add(vec2_t a, vec2_t b) {
    return vec2_new(a.x + b.x, a.y + b.y);
}

vec2_t vec2_sub(vec2_t a, vec2_t b) {
    return vec2_new(a.x - b.x, a.y - b.y);
}

vec2_t vec2_mul(vec2_t v, float factor) {
    return vec2_new(v.x * factor, v.y * factor);
}

vec2_t vec2_div(vec2_t v, float divisor) {
    return vec2_mul(v, 1 / divisor);
}

float vec2_length(vec2_t v) {
    return (float)sqrt(v.x * v.x + v.y * v.y);
}

/*
 * for edge function, see
 * https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/rasterization-stage
 */
float vec2_edge(vec2_t s, vec2_t e, vec2_t v) {
    return (v.x - s.x) * (e.y - s.y) - (v.y - s.y) * (e.x - s.x);
}

void vec2_print(const char *name, vec2_t v) {
    printf("vec2 %s =\n", name);
    printf("    %12f    %12f\n", v.x, v.y);
}

/* vec3 related functions */

vec3_t vec3_new(float x, float y, float z) {
    vec3_t v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

vec3_t vec3_from_vec4(vec4_t v) {
    return vec3_new(v.x, v.y, v.z);
}

vec3_t vec3_min(vec3_t a, vec3_t b) {
    float x = float_min(a.x, b.x);
    float y = float_min(a.y, b.y);
    float z = float_min(a.z, b.z);
    return vec3_new(x, y, z);
}

vec3_t vec3_max(vec3_t a, vec3_t b) {
    float x = float_max(a.x, b.x);
    float y = float_max(a.y, b.y);
    float z = float_max(a.z, b.z);
    return vec3_new(x, y, z);
}

vec3_t vec3_add(vec3_t a, vec3_t b) {
    return vec3_new(a.x + b.x, a.y + b.y, a.z + b.z);
}

vec3_t vec3_sub(vec3_t a, vec3_t b) {
    return vec3_new(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3_t vec3_mul(vec3_t v, float factor) {
    return vec3_new(v.x * factor, v.y * factor, v.z * factor);
}

vec3_t vec3_div(vec3_t v, float divisor) {
    return vec3_mul(v, 1 / divisor);
}

vec3_t vec3_negate(vec3_t v) {
    return vec3_new(-v.x, -v.y, -v.z);
}

float vec3_dot ( vec3_t a, vec3_t b ) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


float vec3_length(vec3_t v) {
    return (float)sqrt(vec3_dot(v, v));
}

vec3_t vec3_normalize(vec3_t v) {
    return vec3_div(v, vec3_length(v));
}


vec3_t vec3_cross(vec3_t a, vec3_t b) {
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return vec3_new(x, y, z);
}

vec3_t vec3_lerp(vec3_t a, vec3_t b, float t) {
    float x = float_lerp(a.x, b.x, t);
    float y = float_lerp(a.y, b.y, t);
    float z = float_lerp(a.z, b.z, t);
    return vec3_new(x, y, z);
}

vec3_t vec3_saturate(vec3_t v) {
    float x = float_saturate(v.x);
    float y = float_saturate(v.y);
    float z = float_saturate(v.z);
    return vec3_new(x, y, z);
}

vec3_t vec3_modulate(vec3_t a, vec3_t b) {
    return vec3_new(a.x * b.x, a.y * b.y, a.z * b.z);
}

void vec3_print(const char *name, vec3_t v) {
    printf("vec3 %s =\n", name);
    printf("    %12f    %12f    %12f\n", v.x, v.y, v.z);
}

/* vec4 related functions */

vec4_t vec4_new(float x, float y, float z, float w) {
    vec4_t v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

vec4_t vec4_from_vec3(vec3_t v, float w) {
    return vec4_new(v.x, v.y, v.z, w);
}

vec4_t vec4_add(vec4_t a, vec4_t b) {
    return vec4_new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

vec4_t vec4_sub(vec4_t a, vec4_t b) {
    return vec4_new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

vec4_t vec4_mul(vec4_t v, float factor) {
    return vec4_new(v.x * factor, v.y * factor, v.z * factor, v.w * factor);
}

vec4_t vec4_div(vec4_t v, float divisor) {
    return vec4_mul(v, 1 / divisor);
}

vec4_t vec4_lerp(vec4_t a, vec4_t b, float t) {
    float x = float_lerp(a.x, b.x, t);
    float y = float_lerp(a.y, b.y, t);
    float z = float_lerp(a.z, b.z, t);
    float w = float_lerp(a.w, b.w, t);
    return vec4_new(x, y, z, w);
}

vec4_t vec4_saturate(vec4_t v) {
    float x = float_saturate(v.x);
    float y = float_saturate(v.y);
    float z = float_saturate(v.z);
    float w = float_saturate(v.w);
    return vec4_new(x, y, z, w);
}

vec4_t vec4_modulate(vec4_t a, vec4_t b) {
    return vec4_new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

void vec4_print(const char *name, vec4_t v) {
    printf("vec4 %s =\n", name);
    printf("    %12f    %12f    %12f    %12f\n", v.x, v.y, v.z, v.w);
}

/* quat related functions */

quat_t quat_new(float x, float y, float z, float w) {
    quat_t q;
    q.x = x;
    q.y = y;
    q.z = z;
    q.w = w;
    return q;
}

float quat_dot(quat_t a, quat_t b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

float quat_length(quat_t q) {
    return (float)sqrt(quat_dot(q, q));
}

quat_t quat_normalize(quat_t q) {
    float factor = 1 / quat_length(q);
    return quat_new(q.x * factor, q.y * factor, q.z * factor, q.w * factor);
}

/*
 * for spherical linear interpolation, see
 * 3D Math Primer for Graphics and Game Development, 2nd Edition, Chapter 8
 */
quat_t quat_slerp(quat_t a, quat_t b, float t) {
    float cos_angle = quat_dot(a, b);
    if (cos_angle < 0) {
        b = quat_new(-b.x, -b.y, -b.z, -b.w);
        cos_angle = -cos_angle;
    }
    if (cos_angle > 1 - EPSILON) {
        float x = float_lerp(a.x, b.x, t);
        float y = float_lerp(a.y, b.y, t);
        float z = float_lerp(a.z, b.z, t);
        float w = float_lerp(a.w, b.w, t);
        return quat_new(x, y, z, w);
    } else {
        float angle = (float)acos(cos_angle);
        float sin_angle = (float)sin(angle);
        float angle_a = (1 - t) * angle;
        float angle_b = t * angle;
        float factor_a = (float)sin(angle_a) / sin_angle;
        float factor_b = (float)sin(angle_b) / sin_angle;
        float x = factor_a * a.x + factor_b * b.x;
        float y = factor_a * a.y + factor_b * b.y;
        float z = factor_a * a.z + factor_b * b.z;
        float w = factor_a * a.w + factor_b * b.w;
        return quat_new(x, y, z, w);
    }
}

void quat_print(const char *name, quat_t q) {
    printf("quat %s =\n", name);
    printf("    %12f    %12f    %12f    %12f\n", q.x, q.y, q.z, q.w);
}

/* mat3 related functions */

mat3_t mat3_identity(void) {
    mat3_t m = {{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		}};
    return m;
}

mat3_t mat3_from_cols(vec3_t c0, vec3_t c1, vec3_t c2) {
    mat3_t m;
    m.m[0][0] = c0.x;
    m.m[1][0] = c0.y;
    m.m[2][0] = c0.z;
    m.m[0][1] = c1.x;
    m.m[1][1] = c1.y;
    m.m[2][1] = c1.z;
    m.m[0][2] = c2.x;
    m.m[1][2] = c2.y;
    m.m[2][2] = c2.z;
    return m;
}

mat3_t mat3_from_mat4(mat4_t m) {
    mat3_t n;
    n.m[0][0] = m.m[0][0];
    n.m[0][1] = m.m[0][1];
    n.m[0][2] = m.m[0][2];
    n.m[1][0] = m.m[1][0];
    n.m[1][1] = m.m[1][1];
    n.m[1][2] = m.m[1][2];
    n.m[2][0] = m.m[2][0];
    n.m[2][1] = m.m[2][1];
    n.m[2][2] = m.m[2][2];
    return n;
}

mat3_t mat3_combine(mat3_t m[4], vec4_t weights_) {
    mat3_t combined = {{{0}}};
    float weights[4];
    int i, r, c;
	
    weights[0] = weights_.x;
    weights[1] = weights_.y;
    weights[2] = weights_.z;
    weights[3] = weights_.w;
	
    for (i = 0; i < 4; i++) {
        float weight = weights[i];
        if (weight > 0) {
            mat3_t source = m[i];
            for (r = 0; r < 3; r++) {
                for (c = 0; c < 3; c++) {
                    combined.m[r][c] += weight * source.m[r][c];
                }
            }
        }
    }
	
    return combined;
}

vec3_t mat3_mul_vec3(mat3_t m, vec3_t v) {
    float product[3];
    int i;
    for (i = 0; i < 3; i++) {
        float a = m.m[i][0] * v.x;
        float b = m.m[i][1] * v.y;
        float c = m.m[i][2] * v.z;
        product[i] = a + b + c;
    }
    return vec3_new(product[0], product[1], product[2]);
}

mat3_t mat3_mul_mat3(mat3_t a, mat3_t b) {
    mat3_t m = {{{0}}};
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                m.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return m;
}

mat3_t mat3_transpose(mat3_t m) {
    mat3_t transpose;
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            transpose.m[i][j] = m.m[j][i];
        }
    }
    return transpose;
}


/*
 * for determinant, adjoint, and inverse, see
 * 3D Math Primer for Graphics and Game Development, 2nd Edition, Chapter 6
 */

static float mat3_determinant(mat3_t m) {
    float a = +m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]);
    float b = -m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]);
    float c = +m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
    return a + b + c;
}

static mat3_t mat3_adjoint(mat3_t m) {
    mat3_t adjoint;
    adjoint.m[0][0] = +(m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2]);
    adjoint.m[0][1] = -(m.m[1][0] * m.m[2][2] - m.m[2][0] * m.m[1][2]);
    adjoint.m[0][2] = +(m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1]);
    adjoint.m[1][0] = -(m.m[0][1] * m.m[2][2] - m.m[2][1] * m.m[0][2]);
    adjoint.m[1][1] = +(m.m[0][0] * m.m[2][2] - m.m[2][0] * m.m[0][2]);
    adjoint.m[1][2] = -(m.m[0][0] * m.m[2][1] - m.m[2][0] * m.m[0][1]);
    adjoint.m[2][0] = +(m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]);
    adjoint.m[2][1] = -(m.m[0][0] * m.m[1][2] - m.m[1][0] * m.m[0][2]);
    adjoint.m[2][2] = +(m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1]);
    return adjoint;
}


mat3_t mat3_inverse_transpose(mat3_t m) {
    mat3_t adjoint, inverse_transpose;
    float determinant, inv_determinant;
    int i, j;
	
    adjoint = mat3_adjoint(m);
    determinant = mat3_determinant(m);
    inv_determinant = 1 / determinant;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            inverse_transpose.m[i][j] = adjoint.m[i][j] * inv_determinant;
        }
    }
    return inverse_transpose;
}


mat3_t mat3_inverse ( mat3_t m ) 
{
    return mat3_transpose(mat3_inverse_transpose(m));
}

void mat3_print(const char *name, mat3_t m) {
    int i, j;
    printf("mat3 %s =\n", name);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("    %12f", m.m[i][j]);
        }
        printf("\n");
    }
}

/* mat4 related functions */

mat4_t mat4_identity(void) {
    mat4_t m = {{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		}};
    return m;
}

mat4_t mat4_from_quat(quat_t q) {
    mat4_t m = mat4_identity();
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float xw = q.x * q.w;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float yw = q.y * q.w;
    float zz = q.z * q.z;
    float zw = q.z * q.w;
	
    m.m[0][0] = 1 - 2 * (yy + zz);
    m.m[0][1] = 2 * (xy - zw);
    m.m[0][2] = 2 * (xz + yw);
	
    m.m[1][0] = 2 * (xy + zw);
    m.m[1][1] = 1 - 2 * (xx + zz);
    m.m[1][2] = 2 * (yz - xw);
	
    m.m[2][0] = 2 * (xz - yw);
    m.m[2][1] = 2 * (yz + xw);
    m.m[2][2] = 1 - 2 * (xx + yy);
	
    return m;
}

mat4_t mat4_mul_mat4(mat4_t a, mat4_t b) {
    mat4_t m = {{{0}}};
    int i, j, k;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            for (k = 0; k < 4; k++) {
                m.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return m;
}

mat4_t mat4_combine(mat4_t m[4], vec4_t weights_) {
    mat4_t combined = {{{0}}};
    float weights[4];
    int i, r, c;
	
    weights[0] = weights_.x;
    weights[1] = weights_.y;
    weights[2] = weights_.z;
    weights[3] = weights_.w;
	
    for (i = 0; i < 4; i++) {
        float weight = weights[i];
        if (weight > 0) {
            mat4_t source = m[i];
            for (r = 0; r < 4; r++) {
                for (c = 0; c < 4; c++) {
                    combined.m[r][c] += weight * source.m[r][c];
                }
            }
        }
    }
	
    return combined;
}

vec4_t mat4_mul_vec4(mat4_t m, vec4_t v) {
    float product[4];
    int i;
    for (i = 0; i < 4; i++) {
        float a = m.m[i][0] * v.x;
        float b = m.m[i][1] * v.y;
        float c = m.m[i][2] * v.z;
        float d = m.m[i][3] * v.w;
        product[i] = a + b + c + d;
    }
    return vec4_new(product[0], product[1], product[2], product[3]);
}


mat4_t mat4_transpose(mat4_t m) {
    mat4_t transpose;
    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            transpose.m[i][j] = m.m[j][i];
        }
    }
    return transpose;
}

/*
 * for determinant, minor, cofactor, adjoint, and inverse, see
 * 3D Math Primer for Graphics and Game Development, 2nd Edition, Chapter 6
 */

static float mat4_minor(mat4_t m, int r, int c) {
    mat3_t cut_down;
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            int row = i < r ? i : i + 1;
            int col = j < c ? j : j + 1;
            cut_down.m[i][j] = m.m[row][col];
        }
    }
    return mat3_determinant(cut_down);
}

static float mat4_cofactor(mat4_t m, int r, int c) {
    float sign = (r + c) % 2 == 0 ? 1.0f : -1.0f;
    float minor = mat4_minor(m, r, c);
    return sign * minor;
}

static mat4_t mat4_adjoint(mat4_t m) {
    mat4_t adjoint;
    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            adjoint.m[i][j] = mat4_cofactor(m, i, j);
        }
    }
    return adjoint;
}



mat4_t mat4_inverse_transpose(mat4_t m) {
    mat4_t adjoint, inverse_transpose;
    float determinant, inv_determinant;
    int i, j;
	
    adjoint = mat4_adjoint(m);
    determinant = 0;
    for (i = 0; i < 4; i++) {
        determinant += m.m[0][i] * adjoint.m[0][i];
    }
    inv_determinant = 1 / determinant;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            inverse_transpose.m[i][j] = adjoint.m[i][j] * inv_determinant;
        }
    }
    return inverse_transpose;
}


mat4_t mat4_inverse(mat4_t m) {
    return mat4_transpose(mat4_inverse_transpose(m));
}

/* transformation matrices */

/*
 * tx, ty, tz: the x, y, and z coordinates of a translation vector
 *
 *  1  0  0 tx
 *  0  1  0 ty
 *  0  0  1 tz
 *  0  0  0  1
 *
 * see http://docs.gl/gl2/glTranslate
 */
mat4_t mat4_translate(float tx, float ty, float tz) {
    mat4_t m = mat4_identity();
    m.m[0][3] = tx;
    m.m[1][3] = ty;
    m.m[2][3] = tz;
    return m;
}

/*
 * sx, sy, sz: scale factors along the x, y, and z axes, respectively
 *
 * sx  0  0  0
 *  0 sy  0  0
 *  0  0 sz  0
 *  0  0  0  1
 *
 * see http://docs.gl/gl2/glScale
 */
mat4_t mat4_scale(float sx, float sy, float sz) {
    mat4_t m = mat4_identity();
    assert(sx != 0 && sy != 0 && sz != 0);
    m.m[0][0] = sx;
    m.m[1][1] = sy;
    m.m[2][2] = sz;
    return m;
}

/*
 * angle: the angle of rotation, in radians
 * vx, vy, vz: the x, y, and z coordinates of a vector, respectively
 *
 * nx*nx*(1-c)+c     ny*nx*(1-c)-s*nz  nz*nx*(1-c)+s*ny  0
 * nx*ny*(1-c)+s*nz  ny*ny*(1-c)+c     nz*ny*(1-c)-s*nx  0
 * nx*nz*(1-c)-s*ny  ny*nz*(1-c)+s*nx  nz*nz*(1-c)+c     0
 * 0                 0                 0                 1
 *
 * nx, ny, nz: the normalized coordinates of the vector, respectively
 * s, c: sin(angle), cos(angle)
 *
 * see http://docs.gl/gl2/glRotate
 */
mat4_t mat4_rotate(float angle, float vx, float vy, float vz) {
    vec3_t n = vec3_normalize(vec3_new(vx, vy, vz));
    float c = (float)cos(angle);
    float s = (float)sin(angle);
    mat4_t m = mat4_identity();
	
    m.m[0][0] = n.x * n.x * (1 - c) + c;
    m.m[0][1] = n.y * n.x * (1 - c) - s * n.z;
    m.m[0][2] = n.z * n.x * (1 - c) + s * n.y;
	
    m.m[1][0] = n.x * n.y * (1 - c) + s * n.z;
    m.m[1][1] = n.y * n.y * (1 - c) + c;
    m.m[1][2] = n.z * n.y * (1 - c) - s * n.x;
	
    m.m[2][0] = n.x * n.z * (1 - c) - s * n.y;
    m.m[2][1] = n.y * n.z * (1 - c) + s * n.x;
    m.m[2][2] = n.z * n.z * (1 - c) + c;
	
    return m;
}

/*
 * angle: the angle of rotation, in radians
 *
 *  1  0  0  0
 *  0  c -s  0
 *  0  s  c  0
 *  0  0  0  1
 *
 * see http://www.songho.ca/opengl/gl_anglestoaxes.html
 */
mat4_t mat4_rotate_x(float angle) {
    float c = (float)cos(angle);
    float s = (float)sin(angle);
    mat4_t m = mat4_identity();
    m.m[1][1] = c;
    m.m[1][2] = -s;
    m.m[2][1] = s;
    m.m[2][2] = c;
    return m;
}

/*
 * angle: the angle of rotation, in radians
 *
 *  c  0  s  0
 *  0  1  0  0
 * -s  0  c  0
 *  0  0  0  1
 *
 * see http://www.songho.ca/opengl/gl_anglestoaxes.html
 */
mat4_t mat4_rotate_y(float angle) {
    float c = (float)cos(angle);
    float s = (float)sin(angle);
    mat4_t m = mat4_identity();
    m.m[0][0] = c;
    m.m[0][2] = s;
    m.m[2][0] = -s;
    m.m[2][2] = c;
    return m;
}

/*
 * angle: the angle of rotation, in radians
 *
 *  c -s  0  0
 *  s  c  0  0
 *  0  0  1  0
 *  0  0  0  1
 *
 * see http://www.songho.ca/opengl/gl_anglestoaxes.html
 */
mat4_t mat4_rotate_z(float angle) {
    float c = (float)cos(angle);
    float s = (float)sin(angle);
    mat4_t m = mat4_identity();
    m.m[0][0] = c;
    m.m[0][1] = -s;
    m.m[1][0] = s;
    m.m[1][1] = c;
    return m;
}

/*
 * eye: the position of the eye point
 * target: the position of the target point
 * up: the direction of the up vector
 *
 * x_axis.x  x_axis.y  x_axis.z  -dot(x_axis,eye)
 * y_axis.x  y_axis.y  y_axis.z  -dot(y_axis,eye)
 * z_axis.x  z_axis.y  z_axis.z  -dot(z_axis,eye)
 *        0         0         0                 1
 *
 * z_axis: normalize(eye-target), the backward vector
 * x_axis: normalize(cross(up,z_axis)), the right vector
 * y_axis: cross(z_axis,x_axis), the up vector
 *
 * see http://www.songho.ca/opengl/gl_camera.html
 */
mat4_t mat4_lookat(vec3_t eye, vec3_t target, vec3_t up) {
    vec3_t z_axis = vec3_normalize(vec3_sub(eye, target));
    vec3_t x_axis = vec3_normalize(vec3_cross(up, z_axis));
    vec3_t y_axis = vec3_cross(z_axis, x_axis);
    mat4_t m = mat4_identity();
	
    m.m[0][0] = x_axis.x;
    m.m[0][1] = x_axis.y;
    m.m[0][2] = x_axis.z;
	
    m.m[1][0] = y_axis.x;
    m.m[1][1] = y_axis.y;
    m.m[1][2] = y_axis.z;
	
    m.m[2][0] = z_axis.x;
    m.m[2][1] = z_axis.y;
    m.m[2][2] = z_axis.z;
	
    m.m[0][3] = -vec3_dot(x_axis, eye);
    m.m[1][3] = -vec3_dot(y_axis, eye);
    m.m[2][3] = -vec3_dot(z_axis, eye);
	
    return m;
}

/*
 * left, right: the coordinates for the left and right clipping planes
 * bottom, top: the coordinates for the bottom and top clipping planes
 * near, far: the distances to the near and far depth clipping planes
 *
 * 2/(r-l)        0         0  -(r+l)/(r-l)
 *       0  2/(t-b)         0  -(t+b)/(t-b)
 *       0        0  -2/(f-n)  -(f+n)/(f-n)
 *       0        0         0             1
 *
 * see http://docs.gl/gl2/glOrtho
 */

#if 0

mat4_t mat4_ortho(float left, float right, float bottom, float top, float near, float far) 
{
    float x_range = right - left;
    float y_range = top - bottom;
    float z_range = far - near;
    mat4_t m = mat4_identity();
    assert(x_range > 0 && y_range > 0 && z_range > 0);
    m.m[0][0] = 2 / x_range;
    m.m[1][1] = 2 / y_range;
    m.m[2][2] = -2 / z_range;
    m.m[0][3] = -(left + right) / x_range;
    m.m[1][3] = -(bottom + top) / y_range;
    m.m[2][3] = -(near + far) / z_range;
    return m;
}

/*
 * left, right: the coordinates for the left and right clipping planes
 * bottom, top: the coordinates for the bottom and top clipping planes
 * near, far: the distances to the near and far depth clipping planes
 *
 * 2n/(r-l)         0   (r+l)/(r-l)           0
 *        0  2n/(t-b)   (t+b)/(t-b)           0
 *        0         0  -(f+n)/(f-n)  -2fn/(f-n)
 *        0         0            -1           0
 *
 * see http://docs.gl/gl2/glFrustum
 */
mat4_t mat4_frustum(float left, float right, float bottom, float top,
                    float near, float far) {
    float x_range = right - left;
    float y_range = top - bottom;
    float z_range = far - near;
    mat4_t m = mat4_identity();
    assert(near > 0 && far > 0);
    assert(x_range > 0 && y_range > 0 && z_range > 0);
    m.m[0][0] = 2 * near / x_range;
    m.m[1][1] = 2 * near / y_range;
    m.m[0][2] = (left + right) / x_range;
    m.m[1][2] = (bottom + top) / y_range;
    m.m[2][2] = -(near + far) / z_range;
    m.m[2][3] = -2 * near * far / z_range;
    m.m[3][2] = -1;
    m.m[3][3] = 0;
    return m;
}

/*
 * right: the coordinates for the right clipping planes (left == -right)
 * top: the coordinates for the top clipping planes (bottom == -top)
 * near, far: the distances to the near and far depth clipping planes
 *
 * 1/r    0         0             0
 *   0  1/t         0             0
 *   0    0  -2/(f-n)  -(f+n)/(f-n)
 *   0    0         0             1
 *
 * this is the same as
 *     float left = -right;
 *     float bottom = -top;
 *     mat4_ortho(left, right, bottom, top, near, far);
 *
 * see http://www.songho.ca/opengl/gl_projectionmatrix.html
 */
mat4_t mat4_orthographic(float right, float top, float near, float far) {
    float z_range = far - near;
    mat4_t m = mat4_identity();
    assert(right > 0 && top > 0 && z_range > 0);
    m.m[0][0] = 1 / right;
    m.m[1][1] = 1 / top;
    m.m[2][2] = -2 / z_range;
    m.m[2][3] = -(near + far) / z_range;
    return m;
}

/*
 * fovy: the field of view angle in the y direction, in radians
 * aspect: the aspect ratio, defined as width divided by height
 * near, far: the distances to the near and far depth clipping planes
 *
 * 1/(aspect*tan(fovy/2))              0             0           0
 *                      0  1/tan(fovy/2)             0           0
 *                      0              0  -(f+n)/(f-n)  -2fn/(f-n)
 *                      0              0            -1           0
 *
 * this is the same as
 *     float half_h = near * (float)tan(fovy / 2);
 *     float half_w = half_h * aspect;
 *     mat4_frustum(-half_w, half_w, -half_h, half_h, near, far);
 *
 * see http://www.songho.ca/opengl/gl_projectionmatrix.html
 */
mat4_t mat4_perspective(float fovy, float aspect, float near, float far) {
    float z_range = far - near;
    mat4_t m = mat4_identity();
    assert(fovy > 0 && aspect > 0);
    assert(near > 0 && far > 0 && z_range > 0);
    m.m[1][1] = 1 / (float)tan(fovy / 2);
    m.m[0][0] = m.m[1][1] / aspect;
    m.m[2][2] = -(near + far) / z_range;
    m.m[2][3] = -2 * near * far / z_range;
    m.m[3][2] = -1;
    m.m[3][3] = 0;
    return m;
}


mat4_t mat4_from_trs(vec3_t t, quat_t r, vec3_t s) 
{
    mat4_t translation = mat4_translate(t.x, t.y, t.z);
    mat4_t rotation = mat4_from_quat(r);
    mat4_t scale = mat4_scale(s.x, s.y, s.z);
    return mat4_mul_mat4(translation, mat4_mul_mat4(rotation, scale));
}


void mat4_print(const char *name, mat4_t m) {
    int i, j;
    printf("mat4 %s =\n", name);
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("    %12f", m.m[i][j]);
        }
        printf("\n");
    }
}

#endif


/* image creating/releasing */

image_t *image_create ( int width, int height, int channels, format_t format ) 
{
    int num_elems = width * height * channels;
    image_t *image;
	
    assert(width > 0 && height > 0 && channels >= 1 && channels <= 4);
    assert(format == LDR || format == HDR);
	
    image = (image_t*)malloc(sizeof(image_t));
    image->format = format;
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->ldr_buffer = NULL;
    image->hdr_buffer = NULL;
	
    if (format == LDR) {
        int size = sizeof(unsigned char) * num_elems;
        image->ldr_buffer = (unsigned char*)malloc(size);
        memset(image->ldr_buffer, 0, size);
    } else {
        int size = sizeof(float) * num_elems;
        image->hdr_buffer = (float*)malloc(size);
        memset(image->hdr_buffer, 0, size);
    }
	
    return image;
}

void image_release(image_t *image)
{
    free(image->ldr_buffer);
    free(image->hdr_buffer);
    free(image);
}

void framebuffer_clear_color(framebuffer_t *framebuffer, vec4_t color) 
{
    int num_pixels = framebuffer->width * framebuffer->height;
    int i;
    for (i = 0; i < num_pixels; i++) {
        framebuffer->color_buffer[i * 4 + 0] = float_to_uchar(color.x);
        framebuffer->color_buffer[i * 4 + 1] = float_to_uchar(color.y);
        framebuffer->color_buffer[i * 4 + 2] = float_to_uchar(color.z);
        framebuffer->color_buffer[i * 4 + 3] = float_to_uchar(color.w);
    }
}

void framebuffer_clear_depth(framebuffer_t *framebuffer, float depth) {
    int num_pixels = framebuffer->width * framebuffer->height;
    int i;
    for (i = 0; i < num_pixels; i++) {
        framebuffer->depth_buffer[i] = depth;
    }
}

framebuffer_t *framebuffer_create ( int width, int height ) 
{
    int color_buffer_size = width * height * 4;
    int depth_buffer_size = sizeof(float) * width * height;
    vec4_t default_color = {0, 0, 0, 1};
    float default_depth = 1;
    framebuffer_t *framebuffer;
	
    assert(width > 0 && height > 0);
	
    framebuffer = (framebuffer_t*)malloc(sizeof(framebuffer_t));
    framebuffer->width = width;
    framebuffer->height = height;
    framebuffer->color_buffer = (unsigned char*)malloc(color_buffer_size);
    framebuffer->depth_buffer = (float*)malloc(depth_buffer_size);
	
    framebuffer_clear_color(framebuffer, default_color);
    framebuffer_clear_depth(framebuffer, default_depth);
	
    return framebuffer;
}

void framebuffer_release(framebuffer_t *framebuffer) {
    free(framebuffer->color_buffer);
    free(framebuffer->depth_buffer);
    free(framebuffer);
}

void private_blit_bgr(framebuffer_t *src, image_t *dst) {
    int width = dst->width;
    int height = dst->height;
    int r, c;
	
    assert(src->width == dst->width && src->height == dst->height);
    assert(dst->format == LDR && dst->channels == 4);
	
    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            int flipped_r = height - 1 - r;
            int src_index = (r * width + c) * 4;
            int dst_index = (flipped_r * width + c) * 4;
            unsigned char *src_pixel = &src->color_buffer[src_index];
            unsigned char *dst_pixel = &dst->ldr_buffer[dst_index];
            dst_pixel[0] = src_pixel[2];  /* blue */
            dst_pixel[1] = src_pixel[1];  /* green */
            dst_pixel[2] = src_pixel[0];  /* red */
        }
    }
}

void private_blit_rgb(framebuffer_t *src, image_t *dst) {
    int width = dst->width;
    int height = dst->height;
    int r, c;
	
    assert(src->width == dst->width && src->height == dst->height);
    assert(dst->format == LDR && dst->channels == 4);
	
    for (r = 0; r < height; r++) {
        for (c = 0; c < width; c++) {
            int flipped_r = height - 1 - r;
            int src_index = (r * width + c) * 4;
            int dst_index = (flipped_r * width + c) * 4;
            unsigned char *src_pixel = &src->color_buffer[src_index];
            unsigned char *dst_pixel = &dst->ldr_buffer[dst_index];
            dst_pixel[0] = src_pixel[0];  /* red */
            dst_pixel[1] = src_pixel[1];  /* green */
            dst_pixel[2] = src_pixel[2];  /* blue */
        }
    }
}

/* misc functions */


static LRESULT CALLBACK process_message ( HWND hwnd, UINT msg, WPARAM wp, LPARAM lp ) 
{
    window_t *window = ( window_t* ) GetProp ( hwnd, WINDOW_ENTRY_NAME );
    if ( window == NULL ) 
    {
        return DefWindowProc ( hwnd, msg, wp, lp );
    }
    else if ( msg == WM_CLOSE ) 
    {
        window->should_close = 1;
        return 0;
    }
    else 
    {
        return DefWindowProc ( hwnd, msg, wp, lp );
    }
}

static void register_class(void) {
    ATOM class_atom;
    WNDCLASS window_class;
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = process_message;
    window_class.cbClsExtra = 0;
    window_class.cbWndExtra = 0;
    window_class.hInstance = GetModuleHandle(NULL);
    window_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    window_class.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    window_class.lpszMenuName = NULL;
    window_class.lpszClassName = WINDOW_CLASS_NAME;
    class_atom = RegisterClass(&window_class);
    assert(class_atom != 0);
    UNUSED_VAR(class_atom);
}

static void unregister_class(void) {
    UnregisterClass(WINDOW_CLASS_NAME, GetModuleHandle(NULL));
}

static void initialize_path(void) {
#ifdef UNICODE
    wchar_t path[MAX_PATH];
    GetModuleFileName(NULL, path, MAX_PATH);
    *wcsrchr(path, L'\\') = L'\0';
    _wchdir(path);
    _wchdir(L"assets");
#else
    char path[MAX_PATH];
    GetModuleFileName(NULL, path, MAX_PATH);
    *strrchr(path, '\\') = '\0';
    _chdir(path);
    _chdir("assets");
#endif
}

void platform_initialize(void) {
    assert(g_initialized == 0);
    register_class();
    initialize_path();
    g_initialized = 1;
}

void platform_terminate(void) {
    assert(g_initialized == 1);
    unregister_class();
    g_initialized = 0;
}

/* window related functions */

static HWND create_window(const char *title_, int width, int height) {
    DWORD style = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT rect;
    HWND handle;
	
#ifdef UNICODE
    wchar_t title[LINE_SIZE];
    mbstowcs(title, title_, LINE_SIZE);
#else
    const char *title = title_;
#endif
	
    rect.left = 0;
    rect.top = 0;
    rect.right = width;
    rect.bottom = height;
    AdjustWindowRect(&rect, style, 0);
    width = rect.right - rect.left;
    height = rect.bottom - rect.top;
	
    handle = CreateWindow(WINDOW_CLASS_NAME, title, style,
                          CW_USEDEFAULT, CW_USEDEFAULT, width, height,
                          NULL, NULL, GetModuleHandle(NULL), NULL);
    assert(handle != NULL);
    return handle;
}

/*
 * for memory device context, see
 * https://docs.microsoft.com/en-us/windows/desktop/gdi/memory-device-contexts
 */
static void create_surface(HWND handle, int width, int height,
                           image_t **out_surface, HDC *out_memory_dc) {
    BITMAPINFOHEADER bi_header;
    HBITMAP dib_bitmap;
    HBITMAP old_bitmap;
    HDC window_dc;
    HDC memory_dc;
    image_t *surface;
	
    surface = image_create(width, height, 4, LDR);
    free(surface->ldr_buffer);
    surface->ldr_buffer = NULL;
	
    window_dc = GetDC(handle);
    memory_dc = CreateCompatibleDC(window_dc);
    ReleaseDC(handle, window_dc);
	
    memset(&bi_header, 0, sizeof(BITMAPINFOHEADER));
    bi_header.biSize = sizeof(BITMAPINFOHEADER);
    bi_header.biWidth = width;
    bi_header.biHeight = -height;  /* top-down */
    bi_header.biPlanes = 1;
    bi_header.biBitCount = 32;
    bi_header.biCompression = BI_RGB;
    dib_bitmap = CreateDIBSection(memory_dc, (BITMAPINFO*)&bi_header,
                                  DIB_RGB_COLORS, (void**)&surface->ldr_buffer,
                                  NULL, 0);
    assert(dib_bitmap != NULL);
    old_bitmap = (HBITMAP)SelectObject(memory_dc, dib_bitmap);
    DeleteObject(old_bitmap);
	
    *out_surface = surface;
    *out_memory_dc = memory_dc;
}

window_t *window_create(const char *title, int width, int height ) 
{
    window_t *window;
    HWND handle;
    image_t *surface;
    HDC memory_dc;
	
    assert(g_initialized && width > 0 && height > 0);
	
    handle = create_window(title, width, height);
    create_surface(handle, width, height, &surface, &memory_dc);
	
    window = (window_t*)malloc(sizeof(window_t));
    memset(window, 0, sizeof(window_t));
    window->handle = handle;
    window->memory_dc = memory_dc;
    window->surface = surface;
	
    SetProp(handle, WINDOW_ENTRY_NAME, window);
    ShowWindow(handle, SW_SHOW);
    return window;
}

void window_destroy(window_t *window)
{
    ShowWindow(window->handle, SW_HIDE);
    RemoveProp(window->handle, WINDOW_ENTRY_NAME);
	
    DeleteDC(window->memory_dc);
    DestroyWindow(window->handle);
	
    window->surface->ldr_buffer = NULL;
    image_release(window->surface);
    free(window);
}

int window_should_close(window_t *window)
{
    return window->should_close;
}

void window_set_userdata(window_t *window, void *userdata) {
    window->userdata = userdata;
}

void *window_get_userdata(window_t *window) {
    return window->userdata;
}

static void present_surface(window_t *window) {
    HDC window_dc = GetDC(window->handle);
    HDC memory_dc = window->memory_dc;
    image_t *surface = window->surface;
    int width = surface->width;
    int height = surface->height;
    BitBlt(window_dc, 0, 0, width, height, memory_dc, 0, 0, SRCCOPY);
    ReleaseDC(window->handle, window_dc);
}

void window_draw_buffer(window_t *window, framebuffer_t *buffer) {
    private_blit_bgr(buffer, window->surface);
    present_surface(window);
}

/* input related functions */

void input_poll_events(void) {
    MSG message;
    while (PeekMessage(&message, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }
}


void input_query_cursor(window_t *window, float *xpos, float *ypos) {
    POINT point;
    GetCursorPos(&point);
    ScreenToClient(window->handle, &point);
    *xpos = (float)point.x;
    *ypos = (float)point.y;
}


/* misc platform functions */

static double get_native_time(void) {
    static double period = -1;
    LARGE_INTEGER counter;
    if (period < 0) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        period = 1 / (double)frequency.QuadPart;
    }
    QueryPerformanceCounter(&counter);
    return counter.QuadPart * period;
}

float platform_get_time(void) {
    static double initial = -1;
    if (initial < 0) {
        initial = get_native_time();
    }
    return (float)(get_native_time() - initial);
}


int main ( int argc, char **argv )
{
	
    int w = 0;
    int h = 0;
    int c = 0;
	
	if ( argc < 2 ) 
	{
		fprintf ( stderr, "Usage: %s image", argv [ 0 ] );
		return -1;
	}
	
    platform_initialize ( );
	
	stbi_set_flip_vertically_on_load ( 1 );
	
	unsigned char *data = stbi_load ( argv [ 1 ], &w, &h, &c, 4 );
	window_t *window = window_create ( "Image Processing", w, h );
	
	
	
#if 0
	
	for ( int r = 0; r < h; r++ )
	{
		for ( c = 0; c < w; c++ )
		{
			int i = ( r * w + c ) * 4;
			
			unsigned char r = data [ i + 0 ];
			unsigned char g = data [ r + 1 ];
			unsigned char b = data [ r + 2 ];
			
			unsigned char gray   = ( unsigned char ) ( ( r * 0.3f + g * 0.58f + b * 0.11f ) );
			data [ i + 0 ] = gray;
			data [ i + 1 ] = gray;
			data [ i + 2 ] = gray;
			data [ i + 3 ] = gray;
			
			
		}
	}
	
#endif
	
	
	// Red channel 
	for ( int r = 0; r < h; r++ )
	{
		for ( c = 0; c < w; c++ )
		{
			int i = ( r * w + c ) * 4;
			
			unsigned char r = data [ i + 0 ];
			unsigned char g = data [ i + 1 ];
			unsigned char b = data [ i + 2 ];
			
			unsigned char c = ( r + g + b ) / 3;
			
			data [ i + 0 ] = c;
			data [ i + 1 ] = c;
			data [ i + 2 ] = c;
			data [ i + 3 ] = 0;
			
			
		}
	}
	
	
#if 0
	// Logarithm Operator
	for ( int r = 0; r < h; r++ )
	{
		for ( c = 0; c < w; c++ )
		{
			int i = ( r * w + c ) * 4;
			
			unsigned char r = data [ i + 0 ];
			unsigned char g = data [ r + 1 ];
			unsigned char b = data [ r + 2 ];
			
			// unsigned char mx = max ( r , max ( g , b ) );
			// unsigned char c = 255 / ( 1 + log ( mx ) );
			
			//data [ i + 0 ] = 255 / ( 1 + log ( r ) );
			//data [ i + 1 ] = 255 / ( 1 + log ( g ) );
			//data [ i + 2 ] = 255 / ( 1 + log ( b ) );
			//data [ i + 3 ] = 0;
			
			
		}
	}
#endif
	
	framebuffer_t *fb = framebuffer_create ( w, h );
	
	memcpy ( fb->color_buffer, data, ( w * h * 4 ) );
	
	
	while ( !window_should_close ( window ) )
	{
		window_draw_buffer ( window, fb );
		
		input_poll_events ( );
	}
	
	
	framebuffer_release ( fb );
	stbi_image_free ( data );
	window_destroy ( window );
	platform_terminate ( );
	
	return 0;
}