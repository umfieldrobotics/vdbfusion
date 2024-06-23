// MIT License
//
// # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fmt/core.h>
#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <openvdb/openvdb.h>
#include <vdbfusion/VDBVolume.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <string>
 

#include "datasets/KITTIOdometry.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"
// #include "nanovdb/examples/ex_raytrace_level_set/nanovdb.cu"
#include <nanovdb/util/Ray.h> 
#include "common.h"
#include <nanovdb/util/HDDA.h>

#include <nanovdb/util/IO.h>

#include "open3d/Open3D.h"

  // Now iterate through the map and save each mesh
//-------modifiction 
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <sstream>
#include <vector>
#include <chrono>

//------------------------------

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::CudaDeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer, int index, const datasets::KITTIDataset::Point &origin)
{ 
    using GridT = nanovdb::FloatGrid;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    auto* h_grid = handle.grid<float>();
    if (!h_grid)
        throw std::runtime_error("GridHandle does not contain a valid host grid");

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

    float              wBBoxDimZ = (float)h_grid->worldBBox().dim()[2] * 2;
    Vec3T              wBBoxCenter = Vec3T(h_grid->worldBBox().min() + h_grid->worldBBox().dim() * 0.5f);
    nanovdb::CoordBBox treeIndexBbox = h_grid->tree().bbox();
    std::cout << "Bounds: "
              << "[" << treeIndexBbox.min()[0] << "," << treeIndexBbox.min()[1] << "," << treeIndexBbox.min()[2] << "] -> ["
              << treeIndexBbox.max()[0] << "," << treeIndexBbox.max()[1] << "," << treeIndexBbox.max()[2] << "]" << std::endl;

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp     compositeOp;

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox, wBBoxDimZ, &origin] __hostdev__(int start, int end, float* image, const GridT* grid) {
        // get an accessor.
        auto acc = grid->tree().getAccessor();

        for (int i = start; i < end; ++i) {
            Vec3T rayEye; 
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);


            //modification--------------------------------------------------------
            //chang the ray direction from nagative z direction to the positive x direction 
            double rotationMatrix[3][3] = {
                {0, 0, 1},
                {1, 0, 0},
                {0, 1, 0}
            };
            double x = rayDir[0];
            double y = rayDir[1];
            double z = rayDir[2];
            rayDir[0] = rotationMatrix[0][0] * x + rotationMatrix[0][1] * y + rotationMatrix[0][2] * z;
            rayDir[1] = rotationMatrix[1][0] * x + rotationMatrix[1][1] * y + rotationMatrix[1][2] * z;
            rayDir[2] = rotationMatrix[2][0] * x + rotationMatrix[2][1] * y + rotationMatrix[2][2] * z;
 
            rayEye[0] = origin(0); 
            rayEye[1] = origin(1); 
            rayEye[2] = origin(2); 

            //modification--------------------------------------------------------

            // generate ray.
            RayT wRay(rayEye, rayDir);
            
             
        
            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndexF(*grid);
            // intersect...
            float  t0;
            CoordT ijk;
            float  v;
            if (nanovdb::ZeroCrossing(iRay, acc, ijk, v, t0)) {
                // write distance to surface. (we assume it is a uniform voxel)
                float wT0 = t0 * float(grid->voxelSize()[0]);
                compositeOp(image, i, width, height, wT0 / (wBBoxDimZ * 2), 1.0f);
            } else {
                // write background value.
                compositeOp(image, i, width, height, 0.0f, 0.0f);
 
            }
        }
    };

    {   

        auto start3 = std::chrono::high_resolution_clock::now();
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(false, renderOp, width, height, h_outImage, h_grid);
            //std::cout << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Host) = " << durationAvg << " ms" << std::endl;

        std::ostringstream filename;
        filename << "loop_output" << index << ".pfm";

        saveImage(filename.str(), width, height, (float*)imageBuffer.data());
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
        std::cout << "save the image took: " << elapsed3.count() << " ms" << std::endl;

    }

#if defined(NANOVDB_USE_CUDA)
    handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");

    imageBuffer.deviceUpload();
    float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(true, renderOp, width, height, d_outImage, d_grid);
            //std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Cuda) = " << durationAvg << " ms" << std::endl;

        imageBuffer.deviceDownload();
        saveImage("raytrace_level_set-nanovdb-cuda.pfm", width, height, (float*)imageBuffer.data());
    }
#endif
}

// // nanovdb.cu function--------------------------------------------------------------------------







// Namespace aliases
using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("KITTIPipeline");
    argparser.add_argument("kitti_root_dir").help("The full path to the KITTI dataset");
    argparser.add_argument("mesh_output_dir").help("Directory to store the resultant mesh");
    argparser.add_argument("--sequence").help("KITTI Sequence");
    argparser.add_argument("--config")
        .help("Dataset specific config file")
        .default_value<std::string>("config/kitti.yaml")
        .action([](const std::string& value) { return value; });
    argparser.add_argument("--n_scans")
        .help("How many scans to map")
        .default_value(int(-1))
        .action([](const std::string& value) { return std::stoi(value); });

    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Invalid Arguments " << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << argparser;
        exit(0);
    }

    auto kitti_root_dir = argparser.get<std::string>("kitti_root_dir"); 
    if (!fs::exists(kitti_root_dir)) {
        std::cerr << kitti_root_dir << "path doesn't exists" << std::endl;
        exit(1);
    }
    
    std::cout<< " the direction: " << kitti_root_dir << std::endl;
    return argparser;
}
}  // namespace










 



extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer);










int main(int argc, char* argv[]) {

    std::cout << "test message to see if the file has been runed" << std::endl;
    auto argparser = ArgParse(argc, argv);

    // VDBVolume configuration
    auto vdbfusion_cfg = 
        vdbfusion::VDBFusionConfig::LoadFromYAML(argparser.get<std::string>("--config")); 
    // Dataset specific configuration
    auto kitti_cfg = datasets::KITTIConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    openvdb::initialize(); 

    // Kitti stuff
    auto n_scans = argparser.get<int>("--n_scans");
    auto kitti_root_dir = fs::path(argparser.get<std::string>("kitti_root_dir"));
    auto sequence = argparser.get<std::string>("--sequence"); 

    // Initialize dataset 
    const auto dataset =
        datasets::KITTIDataset(kitti_root_dir, sequence, n_scans, kitti_cfg.apply_pose_,
                               kitti_cfg.preprocess_, kitti_cfg.min_range_, kitti_cfg.max_range_);

    fmt::print("Integrating {} scans\n", dataset.size());
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_);
    timers::FPSTimer<10> timer;
    //modification---------------------------------------------------------------------------------------
    int index = 0; 
    //modification---------------------------------------------------------------------------------------
    for (const auto& [scan, semantics, origin] : iterable(dataset)) {
        timer.tic();
        tsdf_volume.Integrate(scan, semantics, origin, [](float /*unused*/) { return 1.0; }); //<<------------------------------core  & origionis the camera pose 
        //-----------------------------------------------------------------------------modification 
        auto start0 = std::chrono::high_resolution_clock::now();
        std::cout << " the origin is: " << origin << std::endl;
        std::cout << "enter the loop run part! #" << index <<std::endl;
        const int numIterations = 50;
        const int width = 1024;
        const int height = 1024;
        BufferT   imageBuffer;
        imageBuffer.init(width * height * sizeof(float));
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed0 = end0 - start0;
        std::cout << "create the image buffer took: " << elapsed0.count() << " ms" << std::endl; // 0.06 ms

        auto start1 = std::chrono::high_resolution_clock::now();
        openvdb::FloatGrid::Ptr openvdbGrid = tsdf_volume.tsdf_;

        openvdb::CoordBBox bbox;
        if (openvdbGrid->tree().evalActiveVoxelBoundingBox(bbox)) {
            // Print the dimensions of the bounding box
            openvdb::Coord dim = bbox.dim();
            std::cout << "Bounding box dimensions: " << dim.x() << " x " << dim.y() << " x " << dim.z() << std::endl;
        } else {
            std::cout << "No active voxels in the grid." << std::endl;
        }


        // // try to change some part to be 0 and see what will happen
        // openvdb::FloatGrid::Accessor accessor = openvdbGrid->getAccessor();
        // for(int x = 0; x < 100; x++){
        //     for(int y = 0; y < 100; y++){
        //         for (int z = 0; z < 100; z++){
        //             openvdb::Coord xyz(x, y, z);  // x, y, and z are the indices of the voxel
        //             float newValue = 0.0f;        // the new value you want to set
        //             accessor.setValue(xyz, newValue); // set the new value at the voxel location
        //         }
        //     }
        // }

        // auto [vertices, triangles, labels] = // labels belong to triangles
        //     tsdf_volume.ExtractTriangleMesh(vdbfusion_cfg.fill_holes_, vdbfusion_cfg.min_weight_);

        // for(int i = 0; i < 100; i ++){
        //     std::cout<< " the "  << i << " element is: " << labels[i] << std::endl;
        // }
 
        // std::cout<< " the size of the vector " << labels.size() << std::endl;


        
         

        nanovdb::GridHandle<nanovdb::HostBuffer> handle = nanovdb::openToNanoVDB<nanovdb::HostBuffer>(*openvdbGrid);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
        std::cout << "Conversion to NanoVDB took: " << elapsed1.count() << " ms" << std::endl; //13 ms

        auto start2 = std::chrono::high_resolution_clock::now();
        runNanoVDB(handle, numIterations, width, height, imageBuffer,index, origin);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
        std::cout << "nano rendering took" << elapsed2.count() << " ms" << std::endl; // 4000 ms
        index++;
        //-----------------------------------------------------------------------------modification 
        timer.toc();
    }
    // example/cpp/dataset/KITTIO... / last part helper function defined. 

 


 

    // Store the grid results to disks
    std::string map_name = fmt::format("{out_dir}/kitti_{seq}_{n_scans}_scans",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "seq"_a = sequence, "n_scans"_a = n_scans);
    {
        timers::ScopeTimer timer("Writing VDB grid to disk");
        auto tsdf_grid = tsdf_volume.tsdf_;
        std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
        openvdb::io::File(filename).write({tsdf_grid});
    }

    // Run marching cubes and save a .ply file
    {
        // for semantics, we will save one mesh for each label--28 in total. the label will be associated with the face. TODO discuss this further
        // triangles[0-2] is for map, triangles[3] is label?
        timers::ScopeTimer timer("Writing Mesh to disk");
        auto [vertices, triangles, labels] = // labels belong to triangles
            tsdf_volume.ExtractTriangleMesh(vdbfusion_cfg.fill_holes_, vdbfusion_cfg.min_weight_);

        // TODO: Fix this!
        Eigen::MatrixXd V(vertices.size(), 3);
        for (size_t i = 0; i < vertices.size(); i++) {
            V.row(i) = Eigen::VectorXd::Map(&vertices[i][0], vertices[i].size());
        }

        // TODO: Also this
        // We want a list of matrices, one for each label
        std::map<int, Eigen::MatrixXi> tri_map; // label: (matrix, #elements)
        std::map<int, int32_t> tri_map_sizes; // label: (matrix, #elements)

        for (size_t i = 0; i < triangles.size(); i++) {
            if (tri_map.find(labels[i]) == tri_map.end()) {
                Eigen::MatrixXi F(triangles.size(), 3);
                tri_map[labels[i]] = F;
                tri_map_sizes[labels[i]] = 0;
            }
            auto curr_num_triangles = tri_map_sizes[labels[i]];
            tri_map[labels[i]].row(curr_num_triangles) = Eigen::VectorXi::Map(&triangles[i][0], triangles[i].size());
            tri_map_sizes[labels[i]]++;
        }

        // Now iterate through the map and save each mesh
        for (auto const& [class_label, matrix] : tri_map) {
            std::string filename = fmt::format("{map_name}_{class_label}.ply", fmt::arg("map_name", map_name), fmt::arg("class_label", class_label));
            // truncate matrix so it's just the number of triangles
            auto num_triangles = tri_map_sizes[class_label];
            Eigen::MatrixXi new_matrix(num_triangles, 3);
            for (size_t i = 0; i < num_triangles; i++) {
                new_matrix.row(i) = matrix.row(i);
            }
            igl::write_triangle_mesh(filename, V, new_matrix, igl::FileEncoding::Ascii);
        }
        
    }



    return 0;
}
