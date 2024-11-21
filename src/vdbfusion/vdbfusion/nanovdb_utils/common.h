// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/GridBuilder.h>
#include "ComputePrimitives.h"

// #define NANOVDB_USE_CUDA

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

using LabelGridT = nanovdb::UInt16Grid;

#ifdef __cplusplus
extern "C" {
#endif

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, nanovdb::GridHandle<BufferT>& label_handle, int width, int height, BufferT& imageBuffer, int index, const std::vector<double> origin);

#ifdef __cplusplus
}
#endif

inline __hostdev__ uint32_t CompactBy1(uint32_t x)
{
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

inline __hostdev__ uint32_t SeparateBy1(uint32_t x)
{
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

inline __hostdev__ void mortonDecode(uint32_t code, uint32_t& x, uint32_t& y)
{
    x = CompactBy1(code);
    y = CompactBy1(code >> 1);
}

inline __hostdev__ void mortonEncode(uint32_t& code, uint32_t x, uint32_t y)
{
    code = SeparateBy1(x) | (SeparateBy1(y) << 1);
}

template<typename RenderFn, typename GridT>
inline float renderImage(bool useCuda, const RenderFn renderOp, int width, int height, float* image, const GridT* grid, const LabelGridT* label_grid)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    computeForEach(
        useCuda, width * height, 512, __FILE__, __LINE__, [renderOp, image, grid, label_grid] __hostdev__(int start, int end) {
            renderOp(start, end, image, grid, label_grid);
        });
    computeSync(useCuda, __FILE__, __LINE__);

    auto t1 = ClockT::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    return duration;
}

inline void saveImage(const std::string& filename, int width, int height, const float* image)
{
    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };

    float scale = 1.0f;
    if (isLittleEndian())
        scale = -scale;

    std::fstream fs(filename, std::ios::out | std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    fs << "PF\n"
       << width << "\n"
       << height << "\n"
       << scale << "\n";

    for (int i = 0; i < width * height; ++i) {
        float r1 = image[i];
        float r2 = image[width*height + i];
        float r3 = image[2*width*height + i];
        fs.write((char*)&r1, sizeof(float));
        fs.write((char*)&r2, sizeof(float));
        fs.write((char*)&r3, sizeof(float));
    }
}

template<typename Vec3T>
struct RayGenOp
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;

    inline RayGenOp(float wBBoxDimZ, Vec3T wBBoxCenter)
        : mWBBoxDimZ(wBBoxDimZ)
        , mWBBoxCenter(wBBoxCenter)
    {
    }

    inline __hostdev__ void operator()(int i, int w, int h, Vec3T& outOrigin, Vec3T& outDir) const
    {
        // perspective camera along Z-axis...
        uint32_t x, y;
#if 0
        mortonDecode(i, x, y);
#else
        x = i % w;
        y = i / w;
#endif
        const float fov = 45.f;
        const float u = (float(x) + 0.5f) / w;
        const float v = (float(y) + 0.5f) / h;
        const float aspect = w / float(h);
        const float Px = (2.f * u - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f) * aspect;
        const float Py = (2.f * v - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f);
        const Vec3T origin = mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
        Vec3T       dir(Px, Py, -1.f);
        dir.normalize();
        outOrigin = origin;
        outDir = dir;
    }
};

struct RGB {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct CompositeOp
{
    int semanticColorMap[3*31] = {
        0, 0, 0,     // 0: Black 0 IS FLAG FOR NO CLASS
        0, 255, 0,   // 1: Green
        0, 0, 255,   // 2: Blue
        255, 255, 0, // 3: Yellow
        0, 255, 255, // 4: Cyan
        255, 0, 255, // 5: Magenta
        192, 192, 192, // 6: Silver
        128, 128, 128, // 7: Gray
        128, 0, 0,     // 8: Maroon
        128, 0, 128,   // 9: Purple
        128, 128, 0,   // 10: Olive
        0, 128, 0,     // 11: Dark Green
        0, 128, 128,   // 12: Teal
        0, 0, 128,     // 13: Navy
        255, 165, 0,   // 14: Orange
        255, 192, 203, // 15: Pink
        139, 69, 19,   // 16: Brown
        75, 0, 130,    // 17: Indigo
        173, 255, 47,  // 18: Green Yellow
        240, 230, 140, // 19: Khaki
        255, 105, 180, // 20: Hot Pink
        72, 209, 204,  // 21: Medium Turquoise
        240, 128, 128, // 22: Light Coral
        255, 228, 181, // 23: Moccasin
        152, 251, 152, // 24: Pale Green
        135, 206, 250, // 25: Light Sky Blue
        238, 130, 238, // 26: Violet
        210, 105, 30,  // 27: Chocolate
        0, 191, 255,   // 28: Deep Sky Blue
        255, 250, 205, // 29: Lemon Chiffon
        124, 228, 0};  // 30: Lawn Green

    int instanceColorMap[3*44] = {
        0, 0, 0,
        255, 255, 0,
        28, 230, 255,
        255, 52, 255,
        255, 74, 70,
        0, 137, 65,
        0, 111, 166,
        163, 0, 89,
        255, 219, 229,
        122, 73, 0,
        0, 0, 166,
        99, 255, 172,
        183, 151, 98,
        0, 77, 67,
        143, 176, 255,
        153, 125, 135,
        90, 0, 7,
        128, 150, 147,
        254, 255, 230,
        27, 68, 0,
        79, 198, 1,
        59, 93, 255,
        74, 59, 83,
        255, 47, 128,
        97, 97, 90,
        186, 9, 0,
        107, 121, 0,
        0, 194, 160,
        255, 170, 146,
        255, 144, 201,
        185, 3, 170,
        209, 97, 0,
        221, 239, 255,
        0, 0, 53,
        123, 79, 75,
        161, 194, 153,
        48, 0, 24,
        10, 166, 216,
        1, 51, 73,
        0, 132, 111,
        55, 33, 1,
        255, 181, 0,
        186, 98, 0};

    inline __hostdev__ void operator()(float* outImage, int i, int w, int h, int label, float alpha) const
    {

        uint32_t x, y;
        int      offset;
#if 0
        mortonDecode(i, x, y);
        offset = x + y * w;
#else
        x = i % w;
        y = i / w;
        offset = i;
#endif

        // checkerboard background...
        const int   mask = 1 << 7;
        const float bg = ((x & mask) ^ (y & mask)) ? 1.0f : 0.5f;

        // decode label (upper 16 bits are instance label, lower 16 bits are semantic label)
        uint16_t inst_label = static_cast<uint16_t>(label >> 16);
        uint16_t sem_label = static_cast<uint16_t>(label & 0xFFFF);

        RGB color;

#if defined(NANOVDB_USE_CUDA)
        // int* d_SCM;
        // cudaMalloc((void**)&d_SCM, 31 * 3 * sizeof(int));
        // cudaMemcpy(d_SCM, semanticColorMap, 31 * 3 * sizeof(int), cudaMemcpyHostToDevice);

        // int* d_ICM;
        // cudaMalloc((void**)&d_ICM, 1024 * 3 * sizeof(int));
        // cudaMemcpy(d_ICM, instanceColorMap, 1024 * 3 * sizeof(int), cudaMemcpyHostToDevice);

        if (inst_label == 0) { // no instance label, so go off of semantics
            color.r = semanticColorMap[(3*(sem_label%31))];
            color.g = semanticColorMap[(3*(sem_label%31))+1];
            color.b = semanticColorMap[(3*(sem_label%31))+2];
        }
        else { // instance label
            color.r = instanceColorMap[(3*(inst_label%44))];
            color.b = instanceColorMap[(3*(inst_label%44))+1];
            color.g = instanceColorMap[(3*(inst_label%44))+2];
        }
#else
        if (inst_label == 0) { // no instance label, so go off of semantics TODO FIX THE MOD
            color.r = semanticColorMap[((3*sem_label)%31)];
            color.g = semanticColorMap[((3*sem_label)%31)+1];
            color.b = semanticColorMap[((3*sem_label)%31)+2];
        }
        else { // instance label
            color.r = instanceColorMap[((3*inst_label)%44)];
            color.g = instanceColorMap[((3*inst_label)%44)+1];
            color.b = instanceColorMap[((3*inst_label)%44)+2];
        }
#endif
//         outImage[offset] = alpha * (color.r / 255.0) + (1.0f - alpha) * bg;
//         outImage[w*h + offset] = alpha * (color.b / 255.0) + (1.0f - alpha) * bg;
//         outImage[2*w*h + offset] = alpha * (color.g / 255.0) + (1.0f - alpha) * bg;

        outImage[offset] = (color.r / 255.0) + (1.0f - alpha);
        outImage[w*h + offset] = (color.b / 255.0) + (1.0f - alpha);
        outImage[2*w*h + offset] = (color.g / 255.0) + (1.0f - alpha);
    }
};
