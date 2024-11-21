/*
 * MIT License
 *
 * # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <functional>
#include <tuple>

namespace vdbfusion {

class VDBVolume {
public:
    VDBVolume(float voxel_size, float sdf_trunc, bool space_carving, int num_semantic_classes);
    ~VDBVolume() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud and its labels into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<uint32_t>& labels,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Matrix4d& extrinsics,
                          const std::function<float(float)>& weighting_function) {
        const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, origin, weighting_function);
    }

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<uint32_t>& labels,
                          const Eigen::Matrix4d& extrinsics,
                          const std::function<float(float)>& weighting_function) {
        const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, labels, origin, weighting_function);
    }

    /// @brief Integrate incoming TSDF grid inside the current volume using the TSDF equations
    void Integrate(openvdb::FloatGrid::Ptr grid,
                   const std::function<float(float)>& weighting_function);

    /// @brief Render volume from viewpoint of origin and save as a pfm
    void Render(const std::vector<double> origin_vec,
                const int index);

    /// @brief Fuse a new given sdf value at the given voxel location, thread-safe
    void UpdateTSDF(const float& sdf,
                    const openvdb::Coord& voxel,
                    const std::function<float(float)>& weighting_function);

    /// @brief Prune TSDF grids, ideal utility to cleanup a D(x) volume before exporting it
    openvdb::FloatGrid::Ptr Prune(float min_weight) const;

    /// @brief Extracts a TriangleMesh as the iso-surface in the actual volume
    [[nodiscard]] std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>, std::vector<int>>
    ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;

public:
    /// OpenVDB Grids modeling the signed distance field and the weight grid
    openvdb::FloatGrid::Ptr tsdf_;
    openvdb::FloatGrid::Ptr weights_;
    openvdb::UInt16Grid::Ptr instances_;
    std::unordered_map<uint16_t, std::vector<uint16_t>> instance_dirichlet_parameters_; // map of instance number to alpha parameter vector of length # of semantic classes

    /// VDBVolume public properties
    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
    int num_semantic_classes_;
};

}  // namespace vdbfusion
