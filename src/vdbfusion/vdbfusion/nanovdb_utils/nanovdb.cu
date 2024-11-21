// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
#define _USE_MATH_DEFINES

#include <cmath>
#include <chrono>

#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/util/GridBuilder.h>

#include "common.h"

#include <opencv2/opencv.hpp>

#if defined(NANOVDB_USE_CUDA)
#include <nanovdb/cuda/DeviceBuffer.h>
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, nanovdb::GridHandle<BufferT>& label_handle, int width, int height, BufferT& imageBuffer, int index, const std::vector<double> origin)
{ 
    using GridT = nanovdb::FloatGrid;
    using LabelGridT = nanovdb::UInt16Grid;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::math::Vec3<RealT>;
    using RayT = nanovdb::math::Ray<RealT>;

    auto* h_grid = handle.grid<float>();
    auto* h_label_grid = label_handle.grid<uint16_t>();
    if (!h_grid)
        throw std::runtime_error("GridHandle does not contain a valid host grid");
    if (!h_label_grid)
        throw std::runtime_error("GridHandle does not contain a valid host label grid");

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

#if defined(NANOVDB_USE_CUDA)
    double* d_origin;
    cudaMalloc((void**)&d_origin, 3 * sizeof(double));
    cudaMemcpy(d_origin, origin.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);
#else
    double d_origin[3] = {origin[0], origin[1], origin[2]};
#endif

    float              wBBoxDimZ = (float)h_grid->worldBBox().dim()[2] * 2;
    Vec3T              wBBoxCenter = Vec3T(h_grid->worldBBox().min() + h_grid->worldBBox().dim() * 0.5f);
    nanovdb::CoordBBox treeIndexBbox = h_grid->tree().bbox();
    std::cout << "Bounds: "
              << "[" << treeIndexBbox.min()[0] << "," << treeIndexBbox.min()[1] << "," << treeIndexBbox.min()[2] << "] -> ["
              << treeIndexBbox.max()[0] << "," << treeIndexBbox.max()[1] << "," << treeIndexBbox.max()[2] << "]" << std::endl;

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp     compositeOp;

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox, wBBoxDimZ, d_origin] __hostdev__(int start, int end, float* image, const GridT* grid, const LabelGridT* label_grid) {
        // get an accessor.
        auto acc = grid->tree().getAccessor();
        auto label_acc = label_grid->tree().getAccessor();

        for (int i = start; i < end; ++i) {
            Vec3T rayEye; 
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);

            // change the ray direction from negative z direction to the positive x direction 
            double rotationMatrix[3][3] = {
                {0, 0, -1},
                {-1, 0, 0},
                {0, 1, 0}
            };

            double x = rayDir[0];
            double y = rayDir[1];
            double z = rayDir[2];
            rayDir[0] = rotationMatrix[0][0] * x + rotationMatrix[0][1] * y + rotationMatrix[0][2] * z;
            rayDir[1] = rotationMatrix[1][0] * x + rotationMatrix[1][1] * y + rotationMatrix[1][2] * z;
            rayDir[2] = rotationMatrix[2][0] * x + rotationMatrix[2][1] * y + rotationMatrix[2][2] * z;

            rayEye[0] = d_origin[0];
            rayEye[1] = d_origin[1];
            rayEye[2] = d_origin[2];

            // generate ray.
            RayT wRay(rayEye, rayDir);

            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndexF(*grid);

            // intersect...
            float  t0;
            CoordT ijk;
            float  v;

            if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, v, t0)) {
                // write distance to surface. (we assume it is a uniform voxel)
                float wT0 = t0 * float(grid->voxelSize()[0]);
                auto label = label_acc.getValue(ijk);
                compositeOp(image, i, width, height, label, 1.0f);
            } else {
                // write background value.
                compositeOp(image, i, width, height, 0, 0.0f);
 
            }
        }
    };

#if defined(NANOVDB_USE_CUDA)
    auto t5 = std::chrono::high_resolution_clock::now();
    handle.deviceUpload();
    label_handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    auto* d_label_grid = label_handle.deviceGrid<uint16_t>();
    if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");
    if (!d_label_grid)
    throw std::runtime_error("GridHandle does not contain a valid device label grid");

    imageBuffer.deviceUpload();
    float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

    auto t6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed5 = t6 - t5;
    std::cout << "Device Upload took: " << elapsed5.count() << " ms" << std::endl;

    {
        float duration = renderImage(true, renderOp, width, height, d_outImage, d_grid, d_label_grid);
        std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;

        auto start3 = std::chrono::high_resolution_clock::now();

        imageBuffer.deviceDownload();

        // std::ostringstream filename;
        // filename << "examples/python/out/pfms/" << "loop_output" << index << ".pfm";

        // Create a cv::Mat of size height x width with 3 channels (CV_32FC3) for storing the image.
        cv::Mat mat(height, width, CV_32FC3);

        auto image = (float*)imageBuffer.data();

        // Populate the cv::Mat with the image data
        for (int i = 0; i < width * height; ++i) {
            int y = height - 1 - (i / width);  // Flip the row index (invert y-axis)
            int x = i % width;

            mat.at<cv::Vec3f>(y, x)[0] = image[2 * width * height + i];  // Blue channel
            mat.at<cv::Vec3f>(y, x)[1] = image[width * height + i];      // Green channel
            mat.at<cv::Vec3f>(y, x)[2] = image[i];                       // Red channel
        }

        mat.convertTo(mat, CV_8UC3, 255);

        cv::imshow("Image", mat);
        cv::waitKey(1);

        // saveImage(filename.str(), width, height, (float*)imageBuffer.data());

        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
        std::cout << "Buffer download and displaying the image took: " << elapsed3.count() << " ms" << std::endl;
    }
#else
    {   
        float duration = renderImage(false, renderOp, width, height, h_outImage, h_grid, h_label_grid);
        std::cout << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;

        // std::ostringstream filename;
        // filename << "examples/python/out/pfms/" << "loop_output" << index << ".pfm";
        // saveImage(filename.str(), width, height, (float*)imageBuffer.data());

        auto start3 = std::chrono::high_resolution_clock::now();

        // Create a cv::Mat of size height x width with 3 channels (CV_32FC3) for storing the image.
        cv::Mat mat(height, width, CV_32FC3);

        auto image = (float*)imageBuffer.data();

        // Populate the cv::Mat with the image data
        for (int i = 0; i < width * height; ++i) {
            int y = height - 1 - (i / width);  // Flip the row index (invert y-axis)
            int x = i % width;

            mat.at<cv::Vec3f>(y, x)[0] = image[2 * width * height + i];  // Blue channel
            mat.at<cv::Vec3f>(y, x)[1] = image[width * height + i];      // Green channel
            mat.at<cv::Vec3f>(y, x)[2] = image[i];                       // Red channel
        }

        mat.convertTo(mat, CV_8UC3, 255);

        cv::imshow("Image", mat);
        cv::waitKey(1);

        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
        std::cout << "Displaying the image took: " << elapsed3.count() << " ms" << std::endl;

    }
#endif
}