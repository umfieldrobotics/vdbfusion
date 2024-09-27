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
#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <vdbfusion/VDBVolume.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "datasets/KITTIOdometry.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"

#include "nanovdb_utils/common.h"
#include <nanovdb/util/Ray.h> 
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include "open3d/Open3D.h"

#include <sstream>
#include <vector>
#include <chrono>


// #define NANOVDB_USE_CUDA

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::CudaDeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif


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


int main(int argc, char* argv[]) {

    // Initialize grid types and point attributes types.
    openvdb::initialize();
    // Create a vector with four point positions.
    std::vector<openvdb::Vec3R> positions;
    positions.push_back(openvdb::Vec3R(0, 1, 0));
    positions.push_back(openvdb::Vec3R(1.5, 3.5, 1));
    positions.push_back(openvdb::Vec3R(-1, 6, -2));
    positions.push_back(openvdb::Vec3R(1.1, 1.25, 0.06));
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions, 1);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int pointsPerVoxel = 8;
    float voxelSize = 0.1; // openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    // Print the voxel-size to cout
    std::cout << "VoxelSize=" << voxelSize << std::endl;
    // Create a transform using this voxel-size.
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);
    // Create a PointDataGrid containing these four points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>(positions, *transform);
    // Set the name of the grid
    grid->setName("Points");
    // Create a VDB file object and write out the grid.
    openvdb::io::File("mypoints.vdb").write({grid});
    // Create a new VDB file object for reading.
    openvdb::io::File newFile("mypoints.vdb");
    // Open the file. This reads the file header, but not any grids.
    newFile.open();
    // Read the grid by name.
    openvdb::GridBase::Ptr baseGrid = newFile.readGrid("Points");
    newFile.close();
    // From the example above, "Points" is known to be a PointDataGrid,
    // so cast the generic grid pointer to a PointDataGrid pointer.
    grid = openvdb::gridPtrCast<openvdb::points::PointDataGrid>(baseGrid);
    openvdb::Index64 count = openvdb::points::pointCount(grid->tree());
    std::cout << "PointCount=" << count << std::endl;
    // Iterate over all the leaf nodes in the grid.
    for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        // Verify the leaf origin.
        std::cout << "Leaf" << leafIter->origin() << std::endl;
        // Extract the position attribute from the leaf by name (P is position).
        const openvdb::points::AttributeArray& array =
            leafIter->constAttributeArray("P");
        // Create a read-only AttributeHandle. Position always uses Vec3f.
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);
        // Iterate over the point indices in the leaf.
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            // Extract the voxel-space position of the point.
            openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
            // Extract the index-space position of the voxel.
            const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
            // Compute the world-space position of the point.
            openvdb::Vec3f worldPosition =
                grid->transform().indexToWorld(voxelPosition + xyz);
            // Verify the index and world-space position of the point
            std::cout << "* PointIndex=[" << *indexIter << "] ";
            std::cout << "WorldPosition=" << worldPosition << std::endl;
        }
    }

    std::vector<openvdb::Vec3R>().swap(positions);

    openvdb::Vec3f worldPosition(1.1, 1.26, 0.06);

    openvdb::Vec3f v = grid->transform().worldToIndex(worldPosition);

    openvdb::Coord voxel(v[0], v[1], v[2]);

    auto leaf = grid->tree().probeLeaf(voxel);

    std::cout << worldPosition << " " << voxel << std::endl;

    const openvdb::points::AttributeArray& array =
            leaf->constAttributeArray("P");
    openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);

    for (auto indexIter = leaf->beginIndexOn(); indexIter; ++indexIter) {
        // Extract the voxel-space position of the point.
        openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
        // Extract the index-space position of the voxel.
        const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
        // Compute the world-space position of the point.
        openvdb::Vec3f worldPosition =
            grid->transform().indexToWorld(voxelPosition + xyz);
        // Verify the index and world-space position of the point
        std::cout << "* PointIndex=[" << *indexIter << "] ";
        std::cout << "WorldPosition=" << worldPosition << std::endl;
    }

    // auto argparser = ArgParse(argc, argv);

    // // VDBVolume configuration
    // auto vdbfusion_cfg = 
    //     vdbfusion::VDBFusionConfig::LoadFromYAML(argparser.get<std::string>("--config")); 
    // // Dataset specific configuration
    // auto kitti_cfg = datasets::KITTIConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    // openvdb::initialize(); 

    // // Kitti stuff
    // auto n_scans = argparser.get<int>("--n_scans");
    // auto kitti_root_dir = fs::path(argparser.get<std::string>("kitti_root_dir"));
    // auto sequence = argparser.get<std::string>("--sequence"); 

    // // Initialize dataset 
    // const auto dataset =
    //     datasets::KITTIDataset(kitti_root_dir, sequence, n_scans, kitti_cfg.apply_pose_,
    //                            kitti_cfg.preprocess_, kitti_cfg.min_range_, kitti_cfg.max_range_);

    // fmt::print("Integrating {} scans\n", dataset.size());
    // vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
    //                                  vdbfusion_cfg.space_carving_);
    // timers::FPSTimer<10> timer;
    // //modification--------------------------------cuda-------------------------------------------------------
    // int index = 0; 
    // //modification---------------------------------------------------------------------------------------
    // for (const auto& [scan, semantics, origin] : iterable(dataset)) {
    //     timer.tic();
    //     tsdf_volume.Integrate(scan, semantics, origin, [](float /*unused*/) { return 1.0; });
    //     timer.toc();

    //     // // Render image and save as pfm
    //     // std::cout << "\nFrame #" << index << std::endl;
    //     // const int numIterations = 50; //  what does this do?
    //     // const int width = 691;
    //     // const int height = 256;

    //     // auto timer_imgbuff0 = std::chrono::high_resolution_clock::now();
    //     // BufferT imageBuffer;
    //     // imageBuffer.init(3 * width * height * sizeof(float)); // needs to be a 3 channel image
    //     // auto timer_imgbuff1 = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double, std::milli> elapsed0 = timer_imgbuff1 - timer_imgbuff0;
    //     // std::cout << "Image buffer creation took: " << elapsed0.count() << " ms" << std::endl;

    //     // auto timer_nanovdbconv0 = std::chrono::high_resolution_clock::now();
    //     // openvdb::FloatGrid::Ptr openvdbGrid = tsdf_volume.tsdf_;
    //     // openvdb::UInt32Grid::Ptr openvdbGridLabels = tsdf_volume.instances_;
    //     // openvdb::CoordBBox bbox;
    //     // // if (openvdbGrid->tree().evalActiveVoxelBoundingBox(bbox)) {
    //     // //     // Print the dimensions of the bounding box
    //     // //     openvdb::Coord dim = bbox.dim();
    //     // //     std::cout << "Bounding box dimensions: " << dim.x() << " x " << dim.y() << " x " << dim.z() << std::endl;
    //     // // } else {
    //     // //     std::cout << "No active voxels in the grid." << std::endl;
    //     // // }
    //     // nanovdb::GridHandle<BufferT> handle = nanovdb::openToNanoVDB<BufferT>(*openvdbGrid);
    //     // nanovdb::GridHandle<BufferT> label_handle = nanovdb::openToNanoVDB<BufferT>(*openvdbGridLabels);

    //     // auto timer_nanovdbconv1 = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double, std::milli> elapsed1 = timer_nanovdbconv1 - timer_nanovdbconv0;
    //     // std::cout << "Conversion to NanoVDB took: " << elapsed1.count() << " ms" << std::endl;

    //     // auto timer_render0 = std::chrono::high_resolution_clock::now();

    //     // std::vector<double> origin_vec = {origin(0), origin(1), origin(2)};
    //     // runNanoVDB(handle, label_handle, numIterations, width, height, imageBuffer, index, origin_vec);

    //     // auto timer_render1 = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double, std::milli> elapsed2 = timer_render1 - timer_render0;
    //     // std::cout << "NanoVDB rendering took: " << elapsed2.count() << " ms" << std::endl;

    //     index++;
    // }


    // // Store the grid results to disks
    // std::string map_name = fmt::format("{out_dir}/kitti_{seq}_{n_scans}_scans",
    //                                    "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
    //                                    "seq"_a = sequence, "n_scans"_a = n_scans);
    // {
    //     timers::ScopeTimer timer("Writing VDB grid to disk");
    //     auto tsdf_grid = tsdf_volume.grid_;
    //     std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
    //     openvdb::io::File(filename).write({tsdf_grid});

    //     // // Store the grids in a file
    //     // auto instance_grid = tsdf_volume.instances_;
    //     // std::string filename_inst = fmt::format("{map_name}_inst.vdb", "map_name"_a = map_name);
    //     // openvdb::io::File(filename_inst).write({instance_grid});
    // }

    // // // Run marching cubes and save a .ply file
    // // {
    // //     // for semantics, we will save one mesh for each label--28 in total. the label will be associated with the face. TODO discuss this further
    // //     // triangles[0-2] is for map, triangles[3] is label?
    // //     timers::ScopeTimer timer("Writing Mesh to disk");
    // //     auto [vertices, triangles, labels] = // labels belong to triangles
    // //         tsdf_volume.ExtractTriangleMesh(vdbfusion_cfg.fill_holes_, vdbfusion_cfg.min_weight_);

    // //     // TODO: Fix this!
    // //     Eigen::MatrixXd V(vertices.size(), 3);
    // //     for (size_t i = 0; i < vertices.size(); i++) {
    // //         V.row(i) = Eigen::VectorXd::Map(&vertices[i][0], vertices[i].size());
    // //     }

    // //     // TODO: Also this
    // //     // We want a list of matrices, one for each label
    // //     std::map<int, Eigen::MatrixXi> tri_map; // label: (matrix, #elements)
    // //     std::map<int, int32_t> tri_map_sizes; // label: (matrix, #elements)

    // //     for (size_t i = 0; i < triangles.size(); i++) {
    // //         if (tri_map.find(labels[i]) == tri_map.end()) {
    // //             Eigen::MatrixXi F(triangles.size(), 3);
    // //             tri_map[labels[i]] = F;
    // //             tri_map_sizes[labels[i]] = 0;
    // //         }
    // //         auto curr_num_triangles = tri_map_sizes[labels[i]];
    // //         tri_map[labels[i]].row(curr_num_triangles) = Eigen::VectorXi::Map(&triangles[i][0], triangles[i].size());
    // //         tri_map_sizes[labels[i]]++;
    // //     }

    // //     // Now iterate through the map and save each mesh
    // //     for (auto const& [class_label, matrix] : tri_map) {
    // //         std::string filename = fmt::format("{map_name}_{class_label}.ply", fmt::arg("map_name", map_name), fmt::arg("class_label", class_label));
    // //         // truncate matrix so it's just the number of triangles
    // //         auto num_triangles = tri_map_sizes[class_label];
    // //         Eigen::MatrixXi new_matrix(num_triangles, 3);
    // //         for (size_t i = 0; i < num_triangles; i++) {
    // //             new_matrix.row(i) = matrix.row(i);
    // //         }
    // //         igl::write_triangle_mesh(filename, V, new_matrix, igl::FileEncoding::Ascii);
    // //     }
        
    // // }



    return 0;
}
