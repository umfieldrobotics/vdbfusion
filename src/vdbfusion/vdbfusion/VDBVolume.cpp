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

#include "VDBVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace {

float ComputeSDF(const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& point,
                 const Eigen::Vector3d& voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

Eigen::Vector3d GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3d(v_wf.x(), v_wf.y(), v_wf.z());
}

}  // namespace

namespace vdbfusion {

VDBVolume::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving /* = false*/)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving) {
    grid_ = openvdb::points::PointDataGrid::create(sdf_trunc_);
    grid_->setName("D(x): signed distance grid");
    grid_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    grid_->setGridClass(openvdb::GRID_LEVEL_SET);
}

void VDBVolume::UpdateTSDF(const float& sdf,
                           const openvdb::Coord& voxel,
                           const std::function<float(float)>& weighting_function) {
    if (sdf > -sdf_trunc_) {
        UpdateVoxel(voxel, sdf, weighting_function);
    }
}

void VDBVolume::Integrate(openvdb::points::PointDataGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto leafIter = grid->tree().beginLeaf(); leafIter.test(); ++leafIter) {
        if (leafIter->pointCount() == 0)
            std::cout << "No points at this voxel" << std::endl;
        else if (leafIter->pointCount() == 1) {
            // Iterator over all points in leaf node (in voxel) -- there should only be 1
            auto indexIter = leafIter->beginIndexAll();
            // Create AttributeHandles for tsdf, weights, and instances
            openvdb::points::AttributeArray& tsdfArray = leafIter->attributeArray("tsdf");
            openvdb::points::AttributeHandle<float> tsdfHandle(tsdfArray);

            const float& sdf = tsdfHandle.get(*indexIter);
            const openvdb::Coord& voxel = indexIter.getCoord();

            this->UpdateTSDF(sdf, voxel, weighting_function);
        }
        else
            std::cout << "More than 1 point at this voxel!" << std::endl;
    }
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<uint32_t>& labels,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    if (labels.empty()) {
        std::cerr << "Semantic labels provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = grid_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        int idx = &point - &points[0];
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*grid_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                UpdateVoxel(voxel, sdf, weighting_function, labels[idx]);
            }
        } while (dda.step());
    });
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = grid_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*grid_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                UpdateVoxel(voxel, sdf, weighting_function);
            }
        } while (dda.step());
    });
}

void VDBVolume::UpdateVoxel(const openvdb::Coord& voxel,
                            const float& sdf,
                            const std::function<float(float)>& weighting_function) {
    auto* leaf = grid_->tree().probeLeaf(voxel);
    auto tree_acc = openvdb::tree::ValueAccessorRW<openvdb::points::PointDataTree> (grid_->tree());
    
    if (!leaf) {
        std::cout << "No point at this voxel" << std::endl;
        // // No point at this voxel, so add one

        // openvdb::Vec3f position = indexIter.getCoord().asVec3d();
        // grid_->tree().setValue(indexIter.getCoord(), position);


    }
    else if (leaf->pointCount() == 1) {
        // Iterator over all points in leaf node (in voxel) -- there should only be 1
        auto indexIter = leaf->beginIndexVoxel(voxel);
        
        // Create AttributeHandles for tsdf, weights, and instances
        openvdb::points::AttributeArray& tsdfArray = leaf->attributeArray("tsdf");
        openvdb::points::AttributeHandle<float> tsdfReadHandle(tsdfArray);
        openvdb::points::AttributeWriteHandle<float> tsdfWriteHandle(tsdfArray);

        openvdb::points::AttributeArray& weightsArray = leaf->attributeArray("weights");
        openvdb::points::AttributeHandle<float> weightsReadHandle(weightsArray);
        openvdb::points::AttributeWriteHandle<float> weightWriteHandle(weightsArray);
        
        float last_tsdf = tsdfReadHandle.get(*indexIter);
        float last_weight = weightsReadHandle.get(*indexIter);

        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);

        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);

        tsdfWriteHandle.set(*indexIter, new_tsdf);
        weightWriteHandle.set(*indexIter, new_weight);
    }
    else
        std::cout << "More than 1 point at this voxel!" << std::endl;
}

void VDBVolume::UpdateVoxel(const openvdb::Coord& voxel,
                            const float& sdf,
                            const std::function<float(float)>& weighting_function,
                            const uint16_t label) {

    for (auto leafIter = grid_->tree().cbeginLeaf(); leafIter; ++leafIter) {
        const openvdb::points::AttributeArray& positionArray =
                leafIter->constAttributeArray("P");
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);
        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            // Extract the voxel-space position of the point.
            openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
            // Extract the world-space position of the voxel.
            openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
            // Compute the world-space position of the point.
            openvdb::Vec3f worldPosition =
                grid_->transform().indexToWorld(voxelPosition + xyz);
            // Extract the radius of the point.
            // Verify the index, world-space position and radius of the point.
            std::cout << "* PointIndex=[" << *indexIter << "] ";
            std::cout << "WorldPosition=" << worldPosition << " ";
        }
    }

    auto* leaf = grid_->tree().probeLeaf(voxel);
    
    if (!leaf)
        // std::cout << "No points at this voxel" << std::endl;
        int a = 2;
    else if (leaf->pointCount() == 1) {
        // Iterator over all points in leaf node (in voxel) -- there should only be 1
        auto indexIter = leaf->beginIndexVoxel(voxel);
        
        // Create AttributeHandles for tsdf, weights, and instances
        openvdb::points::AttributeArray& tsdfArray = leaf->attributeArray("tsdf");
        openvdb::points::AttributeHandle<float> tsdfReadHandle(tsdfArray);
        openvdb::points::AttributeWriteHandle<float> tsdfWriteHandle(tsdfArray);

        openvdb::points::AttributeArray& weightArray = leaf->attributeArray("weights");
        openvdb::points::AttributeHandle<float> weightReadHandle(weightArray);
        openvdb::points::AttributeWriteHandle<float> weightWriteHandle(weightArray);

        openvdb::points::AttributeArray& instanceArray = leaf->attributeArray("instances"); // TODO take this out later
        openvdb::points::AttributeHandle<uint32_t> instanceReadHandle(instanceArray);
        openvdb::points::AttributeWriteHandle<uint32_t> instanceWriteHandle(instanceArray);
        
        float last_tsdf = tsdfReadHandle.get(*indexIter);
        float last_weight = weightReadHandle.get(*indexIter);
        uint16_t last_label = instanceReadHandle.get(*indexIter);

        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);

        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
        const uint16_t new_label = label; // TODO FIX THIS UPDATING SCHEME

        tsdfWriteHandle.set(*indexIter, new_tsdf);
        weightWriteHandle.set(*indexIter, new_weight);
        instanceWriteHandle.set(*indexIter, new_label);
    }
    else
        std::cout << "More than 1 point at this voxel!" << std::endl;
}

openvdb::points::PointDataGrid::Ptr VDBVolume::Prune(float min_weight) const {
    // Iterate through all leaf nodes + points to disable points with weight below threshold, then prune
    for (auto leafIter = grid_->tree().beginLeaf(); leafIter.test(); ++leafIter) {
        if (leafIter->pointCount() == 0)
            std::cout << "No points at this voxel" << std::endl;
        else if (leafIter->pointCount() == 1) {
            // Create AttributeHandles for tsdf and weights
            openvdb::points::AttributeArray& tsdfArray = leafIter->attributeArray("tsdf");
            openvdb::points::AttributeHandle<float> tsdfReadHandle(tsdfArray);
            openvdb::points::AttributeWriteHandle<float> tsdfWriteHandle(tsdfArray);

            openvdb::points::AttributeArray& weightArray = leafIter->attributeArray("weights");
            openvdb::points::AttributeHandle<float> weightReadHandle(weightArray);
            openvdb::points::AttributeWriteHandle<float> weightWriteHandle(weightArray);

            openvdb::points::AttributeArray& activeArray = leafIter->attributeArray("__active");
            openvdb::points::AttributeWriteHandle<float> activeWriteHandle(activeArray);

            // Set each point inactive
            for (auto indexIter = leafIter->beginIndexAll(); indexIter; ++indexIter) {
                const float& last_tsdf = tsdfReadHandle.get(*indexIter);
                const float last_weight = weightReadHandle.get(*indexIter);

                // If TSDF weights are below threshold, set point inactive
                if (last_weight < min_weight) {
                    activeWriteHandle.set(*indexIter, false);
                }
            }
        }
        else
            std::cout << "More than 1 point at this voxel!" << std::endl;
    }
    // Delete inactive points by pruning
    grid_->tree().prune();
}
}  // namespace vdbfusion
