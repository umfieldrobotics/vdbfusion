// MIT License
//
// Copyright (c) 2024 Anja Sheppard, University of Michigan
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

#include "SceneNetOdometry.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "open3d/Open3D.h"

namespace fs = std::filesystem;

namespace {

std::vector<std::string> GetDepthFiles(const fs::path& depth_path, int n_scans) {
    std::vector<std::string> depth_files;
    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".png") {
            depth_files.emplace_back(entry.path().string());
        }
    }
    if (depth_files.empty()) {
        std::cerr << depth_path << "path doesn't have any .png" << std::endl;
        exit(1);
    }
    std::sort(depth_files.begin(), depth_files.end());
    if (n_scans > 0) {
        depth_files.erase(depth_files.begin() + n_scans, depth_files.end());
    }
    return depth_files;
}

std::vector<std::string> GetLabelFiles(const fs::path& label_path, int n_scans) {
    std::vector<std::string> label_files;
    for (const auto& entry : fs::directory_iterator(label_path)) {
        if (entry.path().extension() == ".png") {
            label_files.emplace_back(entry.path().string());
        }
    }
    if (label_files.empty()) {
        std::cerr << label_path << "path doesn't have any .png" << std::endl;
        exit(1);
    }
    std::sort(label_files.begin(), label_files.end());
    if (n_scans > 0) {
        label_files.erase(label_files.begin() + n_scans, label_files.end());
    }
    return label_files;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>> ReadSceneNetDepthAndLabels(const std::string& depth_path, const std::string& label_path, const fs::path& calib_file, float min_range, float max_range) {
    // Read depth
    cv::Mat depth_mat = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    // Read label
    cv::Mat label_mat = cv::imread(label_path, cv::IMREAD_UNCHANGED);

    assert(depth_mat.size() == label_mat.size());

    // Convert cv Mat to vector--figure out better way of doing this
    std::vector<float> depth_data(depth_mat.size().width * depth_mat.size().height); // should only be one channel
    std::vector<int> label_data(label_mat.size().width * label_mat.size().height);

    for (size_t i = 0; i < depth_mat.size().height; i++) {
        for (size_t j = 0; j < depth_mat.size().width; j++) {
            depth_data[depth_mat.size().width*i+j] = float(depth_mat.at<uint16_t>(i, j, 0)) / 1000.0; // SceneNet depth values are in millimeters
            label_data[label_mat.size().width*i+j] = int(label_mat.at<uchar>(i, j));
        }
    }

    // Read in config file parameters TODO move this out of loop
    float fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;

    std::ifstream calib_in(calib_file, std::ios_base::in);

    if (!calib_in.is_open()) {
        std::cerr << "Error: Could not open the file " << calib_file << "\n";
        exit(0);
    }

    std::string line;
    while (std::getline(calib_in, line)) {
        if (line.empty()) continue;

        size_t colonPos = line.find(':');

        std::string key = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "fx") {
            fx = std::stod(value);;
        } else if (key == "fy") {
            fy = std::stod(value);;
        } else if (key == "cx") {
            cx = std::stod(value);;
        } else if (key == "cy") {
            cy = std::stod(value);;
        }
        }

    std::vector<Eigen::Vector3d> pc_points; // store the 3D points for creating the pointcloud
    std::vector<uint32_t> labels; // store the labels that aren't tossed

    int num_rows = depth_mat.size().height;
    int num_cols = depth_mat.size().width;
    
    for (size_t u = 0; u < num_rows; u++) {
        for (size_t v = 0; v < num_cols; v++) {
            size_t i = num_cols * u + v;

            float d = depth_data[i];

            if (d <= 0 || std::isnan(d)) continue;
            if (d > max_range || d < min_range) continue; // Skip if out of range

            float x_norm = (v - cx) / fx;
            float y_norm = (u - cy) / fy;

            // Depth value as Euclidean distance, need to convert
            float norm_squared = x_norm * x_norm + y_norm * y_norm + 1.0f;
            float z = d / std::sqrt(norm_squared);

            float x = x_norm * z;
            float y = y_norm * z;

            Eigen::Vector3d p {x, y, z};

            pc_points.push_back(p);
            labels.push_back(label_data[i]);
        }
    }


    open3d::geometry::PointCloud pc = open3d::geometry::PointCloud(pc_points);
    std::vector<Eigen::Vector3d> points = pc.points_;

    return std::make_tuple(points, labels);
}

void PreProcessCloud(std::vector<Eigen::Vector3d>& points, std::vector<uint32_t>& labels, float min_range, float max_range) {
    bool invert = true;
    std::vector<bool> mask = std::vector<bool>(points.size(), invert);
    size_t pos = 0;
    for (auto & point : points) {
        if (point.norm() > max_range || point.norm() < min_range) {
            mask.at(pos) = false;
        }
        ++pos;
    }
    size_t counter = 0;
    for (size_t i = 0; i < points.size(); i++) {
        if (mask[i]) {
            points.at(counter) = points.at(i);
            labels.at(counter) = labels.at(i);
            ++counter;
        }
    }
    points.resize(counter);
    labels.resize(counter);
}

void TransformPoints(std::vector<Eigen::Vector3d>& points, const Eigen::Matrix4d& transformation) {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
            transformation * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

std::vector<Eigen::Matrix4d> GetGTPoses(const fs::path& poses_file, const fs::path& calib_file, const bool rgbd_) {
    std::vector<Eigen::Matrix4d> poses;
    Eigen::Matrix4d T_cam_velo = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d T_velo_cam = Eigen::Matrix4d::Zero();

    // auxiliary variables to read the txt files
    std::string ss;
    float P_00, P_01, P_02, P_03, P_10, P_11, P_12, P_13, P_20, P_21, P_22, P_23;

    std::ifstream poses_in(poses_file, std::ios_base::in);
    // clang-format off
    while (poses_in >>
            P_00 >> P_01 >> P_02 >> P_03 >>
            P_10 >> P_11 >> P_12 >> P_13 >>
            P_20 >> P_21 >> P_22 >> P_23) {
        Eigen::Matrix4d P;
        P << P_00, P_01, P_02, P_03,
             P_10, P_11, P_12, P_13,
             P_20, P_21, P_22, P_23,
             0.00, 0.00, 0.00, 1.00;
        poses.emplace_back(P);
    }
    // clang-format on
    return poses; // in camera coordinate frame
}

}  // namespace 
namespace datasets {

SceneNetDataset::SceneNetDataset(const std::string& scenenet_root_dir,
                           const std::string& sequence,
                           int n_scans,
                           bool rgbd) {
    rgbd_ = rgbd;
    // TODO: to be completed
    auto scenenet_root_dir_ = fs::absolute(fs::path(scenenet_root_dir));
    auto scenenet_sequence_dir = fs::absolute(fs::path(scenenet_root_dir) / "train/0" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(scenenet_sequence_dir / "poses.txt",
                        scenenet_sequence_dir / "intrinsics.txt", rgbd_);
    scan_files_ = GetDepthFiles(fs::absolute(scenenet_sequence_dir / "depth/"), n_scans);
}

SceneNetDataset::SceneNetDataset(const std::string& scenenet_root_dir,
                           const std::string& sequence,
                           int n_scans,
                           bool apply_pose,
                           bool preprocess,
                           float min_range,
                           float max_range,
                           bool rgbd)
    : apply_pose_(apply_pose),
      preprocess_(preprocess),
      min_range_(min_range),
      max_range_(max_range),
      rgbd_(rgbd) {
    auto scenenet_root_dir_ = fs::absolute(fs::path(scenenet_root_dir));
    scenenet_sequence_dir_ = fs::absolute(fs::path(scenenet_root_dir_) / "train/0" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(scenenet_sequence_dir_ / "poses.txt",
                        scenenet_sequence_dir_ / "intrinsics.txt", rgbd_);
    if(rgbd_) {
        depth_files_ = GetDepthFiles(fs::absolute(scenenet_sequence_dir_ / "depth/"), n_scans);
        label_files_ = GetLabelFiles(fs::absolute(scenenet_sequence_dir_ / "semantic/"), n_scans);
    }
    else {
        std::cout << "ERROR: There are no pointclouds for SceneNet. Please make sure that the rgbd option in the .yaml file is set to True." << std::endl;
    }
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>, Eigen::Matrix4d> SceneNetDataset::operator[](int idx) const {
    if (rgbd_) {
        auto [points, semantics] = ReadSceneNetDepthAndLabels(depth_files_[idx], label_files_[idx], fs::absolute(scenenet_sequence_dir_ / "intrinsics.txt"), min_range_, max_range_);

        // if (preprocess_) PreProcessCloud(points, semantics, min_range_, max_range_); // if this is enabled, the preprocessing will make the length of the laser scan points shorter than the # of labels
        if (apply_pose_) TransformPoints(points, poses_[idx]);
        // const Eigen::Matrix4d origin = poses_[idx].block<3, 1>(0, 3);
        return std::make_tuple(points, semantics, poses_[idx]);
    }
    else {
        std::cerr << "The only datatype option for SceneNet is rgbd!\n";
        exit(0);
    }
}
}  // namespace datasets
