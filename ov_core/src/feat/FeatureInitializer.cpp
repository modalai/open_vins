/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "FeatureInitializer.h"

#include "Feature.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#define RANSAAC_THRESHOLD 0.02f //SUBJECT TO CHANGE 0.5 0.1 

using namespace ov_core;

bool FeatureInitializer::single_triangulation(std::shared_ptr<Feature> feat,
                                              std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {
//JOAO ADDS
//==============================================================================================================
//NOTE: THIS ADDS A SECOND FOR-LOOP FOR COMPUTING THE INLIERS AFTER SOLVING THE LINEAR SYSTEM.
//      ARGUABLY, WE CAN COMPUTE THE QUALITY AS WE BUILD THE LINEAR SYSTEM INSTEAD,
//      BUT THE REPROJECTION ERROR WON'T BE AS GOOD SINCE WE DON'T HAVE THE FINAL FEATURE 3D GLOBAL POSITION YET.
//==============================================================================================================
    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    int total_meas = 0;
    size_t anchor_most_meas = 0;
    size_t most_meas = 0;
    for (auto const &pair : feat->timestamps) {
        total_meas += (int)pair.second.size();
        if (pair.second.size() > most_meas) {
            anchor_most_meas = pair.first;
            most_meas = pair.second.size();
        }
    }
    feat->anchor_cam_id = anchor_most_meas;
    feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back();

    // Our linear system matrices
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
//JOAO ADDS
//=====================================================================================
    //CAN ONLY PICK ONE
    if (_options.raansac_gn && _options.raansac_tri){
        _options.raansac_gn = false;
    }
//=====================================================================================

    // Get the position of the anchor pose
    ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
    const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

    // Loop through each camera for this feature
    for (auto const &pair : feat->timestamps) {

        // Add CAM_I features
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
        
            // Get the position of this clone in the global
            const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
            const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

            // Convert current position relative to anchor
            Eigen::Matrix<double, 3, 3> R_AtoCi;
            R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
            Eigen::Matrix<double, 3, 1> p_CiinA;
            p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

            // Get the UV coordinate normal
            Eigen::Matrix<double, 3, 1> b_i;
            b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
            b_i = R_AtoCi.transpose() * b_i;
            b_i = b_i / b_i.norm();
            Eigen::Matrix3d Bperp = skew_x(b_i);

            // Append to our linear system
            Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
            A += Ai;
            b += Ai * p_CiinA;
        }
    }

    // Solve the linear system
    Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // std::stringstream ss;
    // ss << feat->featid << " - cond " << std::abs(condA) << " - z " << p_f(2, 0) << std::endl;
    // PRINT_DEBUG(ss.str().c_str());

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    if (std::abs(condA) > _options.max_cond_number || p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist ||
        std::isnan(p_f.norm())) {
//        fprintf(stderr, "CHECKS:\n");
//        fprintf(stderr, "cond: %6.5f > %f\n", condA, _options.max_cond_number);
//        fprintf(stderr, "dist: %6.5f < %f or %6.5f > %f\n", p_f(2, 0), _options.min_dist, p_f(2, 0),  _options.max_dist);
//        fprintf(stderr, "nan? norm: %d\n", std::isnan(p_f.norm()));
        return false;
    }

    // Store it in our feature object
    feat->p_FinA = p_f;
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
//JOAO ADDS
//=====================================================================================
    //RANSAAC TEST
    if (_options.raansac_tri){
        int inliers = 0;
        int total_obs = 0;

        for (auto const &pair : feat->timestamps) {
            for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
                
                total_obs++;

                //CLONE POSITION IN G
                const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
                const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

                //CONVERT LATEST Pf INTO CLONE CAMERA FRAME
                Eigen::Matrix<double, 3, 1> p_FinCi = R_GtoCi * (feat->p_FinG - p_CiinG);

                //PROJECT
                Eigen::Vector2d reproj_uv(p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2));

                //OBSERVED
                Eigen::Vector2d observed_uv(feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1));

                //REPROJECTION ERROR
                double reprojection_error = (reproj_uv - observed_uv).norm();

                if (reprojection_error < RANSAAC_THRESHOLD) {
                    inliers++;
                }
            }
        }
        feat->ransac_quality =  static_cast<float>(inliers) / total_obs;
    }
//=====================================================================================
    return true;
}

bool FeatureInitializer::single_triangulation_1d(std::shared_ptr<Feature> feat,
                                                 std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    int total_meas = 0;
    size_t anchor_most_meas = 0;
    size_t most_meas = 0;
    for (auto const &pair : feat->timestamps) {
        total_meas += (int)pair.second.size();
        if (pair.second.size() > most_meas) {
            anchor_most_meas = pair.first;
            most_meas = pair.second.size();
        }
    }
    feat->anchor_cam_id = anchor_most_meas;
    feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back();
    size_t idx_anchor_bearing = feat->timestamps.at(feat->anchor_cam_id).size() - 1;

    // Our linear system matrices
    double A = 0.0;
    double b = 0.0;

    // Get the position of the anchor pose
    ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
    const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

    // Get bearing in anchor frame
    Eigen::Matrix<double, 3, 1> bearing_inA;
    bearing_inA << feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(0),
        feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(1), 1;
    bearing_inA = bearing_inA / bearing_inA.norm();

    // Loop through each camera for this feature
    for (auto const &pair : feat->timestamps) {

        // Add CAM_I features
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

            // Skip the anchor bearing
            if ((int)pair.first == feat->anchor_cam_id && m == idx_anchor_bearing)
                continue;

            // Get the position of this clone in the global
            const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
            const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

            // Convert current position relative to anchor
            Eigen::Matrix<double, 3, 3> R_AtoCi;
            R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
            Eigen::Matrix<double, 3, 1> p_CiinA;
            p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

            // Get the UV coordinate normal
            Eigen::Matrix<double, 3, 1> b_i;
            b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
            b_i = R_AtoCi.transpose() * b_i;
            b_i = b_i / b_i.norm();
            Eigen::Matrix3d Bperp = skew_x(b_i);

            // Append to our linear system
            Eigen::Vector3d BperpBanchor = Bperp * bearing_inA;
            A += BperpBanchor.dot(BperpBanchor);
            b += BperpBanchor.dot(Bperp * p_CiinA);
        }
    }

    // Solve the linear system
    double depth = b / A;
    Eigen::MatrixXd p_f = depth * bearing_inA;

    // Then set the flag for bad (i.e. set z-axis to nan)
    if (p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist || std::isnan(p_f.norm())) {
        return false;
    }

    // Store it in our feature object
    feat->p_FinA = p_f;
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
    return true;
}



bool FeatureInitializer::single_gaussnewton(std::shared_ptr<Feature> feat,
                                            std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {
//JOAO ADD
//=====================================================================================
//NOTE: ADDING RAANSAC HERE CHECKS INLIER METRIC DURING OPTIMIZATION POST-TRIANGULATION
//      IN ESSENCE, FEATURES THAT CONVERGE FASTER ARE LIKELY TO BE INLIERS
//=====================================================================================

    // Get into inverse depth
    double rho = 1 / feat->p_FinA(2);
    double alpha = feat->p_FinA(0) / feat->p_FinA(2);
    double beta = feat->p_FinA(1) / feat->p_FinA(2);

    // Optimization parameters
    double lam = _options.init_lamda;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double, 3, 3> Hess = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> grad = Eigen::Matrix<double, 3, 1>::Zero();

//JOAO ADDS
//=====================================================================================
    //RANSAAC TEST
        int inliers = 0;
        int total_obs = 0;
//=====================================================================================
    // Cost at the last iteration
    double cost_old = compute_error(clonesCAM, feat, alpha, beta, rho);

    // Get the position of the anchor pose
    const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < _options.max_runs && lam < _options.max_lamda && eps > _options.min_dx) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {

            Hess.setZero();
            grad.setZero();

            double err = 0;

            // Loop through each camera for this feature
            for (auto const &pair : feat->timestamps) {

                // Add CAM_I features
                for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

                    //=====================================================================================
                    //=====================================================================================

                    // Get the position of this clone in the global
                    const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).Rot();
                    const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).pos();
                    // Convert current position relative to anchor
                    Eigen::Matrix<double, 3, 3> R_AtoCi;
                    R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
                    Eigen::Matrix<double, 3, 1> p_CiinA;
                    p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
                    Eigen::Matrix<double, 3, 1> p_AinCi;
                    p_AinCi.noalias() = -R_AtoCi * p_CiinA;

                    //=====================================================================================
                    //=====================================================================================

                    // Middle variables of the system
                    double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
                    double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
                    double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
                    // Calculate jacobian
                    double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    Eigen::Matrix<double, 2, 3> H;
                    H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
                    // Calculate residual
                    Eigen::Matrix<float, 2, 1> z;
                    z << hi1 / hi3, hi2 / hi3;
                    Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;

                    //=====================================================================================
                    //=====================================================================================

                    // Append to our summation variables
                    err += std::pow(res.norm(), 2);
                    grad.noalias() += H.transpose() * res.cast<double>();
                    Hess.noalias() += H.transpose() * H;

                //JOAO ADDS
                //=====================================================================================
                    //RANSAAC THRESHOLD
                    if (_options.raansac_gn){
                        total_obs++;
                        if (res.norm() < RANSAAC_THRESHOLD){
                            inliers++;
                        }
                    }
                //=====================================================================================
                }
            }
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double, 3, 3> Hess_l = Hess;
        for (size_t r = 0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r, r) *= (1.0 + lam);
        }

        Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = compute_error(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));

        // Debug print
        // std::stringstream ss;
        // ss << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;
        // PRINT_DEBUG(ss.str().c_str());

        // Check if converged
        if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step
        // Else inflate lambda (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam / _options.lam_mult;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam * _options.lam_mult;
            continue;
        }
    }

    // Revert to standard, and set to all
    feat->p_FinA(0) = alpha / rho;
    feat->p_FinA(1) = beta / rho;
    feat->p_FinA(2) = 1 / rho;

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    for (auto const &pair : feat->timestamps) {
        // Loop through the other clones to see what the max baseline is
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
            // Get the position of this clone in the global
            const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
            // Convert current position relative to anchor
            Eigen::Matrix<double, 3, 1> p_CiinA = R_GtoA * (p_CiinG - p_AinG);
            // Dot product camera pose and nullspace
            double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
            if (base_line > base_line_max)
                base_line_max = base_line;
        }
    }
    // std::stringstream ss;
    // ss << feat->featid << " - max base " << (feat->p_FinA.norm() / base_line_max) << " - z " << feat->p_FinA(2) << std::endl;
    // PRINT_DEBUG(ss.str().c_str());

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist ||
        (feat->p_FinA.norm() / base_line_max) > _options.max_baseline || std::isnan(feat->p_FinA.norm())) {
        return false;
    }

    // Finally get position in global frame
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;

//JOAO ADDS
    //=====================================================================================
    //NOTE: THIS ONLY ADDS THE LATEST RAANSAC QUALITY METRIC
    //      WE COULD HAVE A HISTORY OF RAANSAC QUALITY METRICS FOR EACH FEATURE,
    //      BUT I DON'T THINK IT'S NECESSARY
    if (_options.raansac_gn){
    //ADD RANSAC QUALITY METRIC
        feat->ransac_quality =  static_cast<float>(inliers) / total_obs;
    }
    //=====================================================================================
    return true;
}

double FeatureInitializer::compute_error(std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                         std::shared_ptr<Feature> feat, double alpha, double beta, double rho) {
    // Total error
    double err = 0;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // MAI NOTE
    // this function is being used to keep track of the relative error of our depth measurements
    // on top of the original open vins application (feature refinement metric)
    // because of this, extra safety checks have been added:
    // making sure the anchor cam / clone fields have been set properly
    // making sure that we have fully recorded the measurement at that time in clones cam
    /////////////////////////////////////////////////////////////////////////////////////////////


    // cant use anchor cam id if we dont have one..
    if (feat->anchor_cam_id == -1 || feat->anchor_clone_timestamp == -1){
        // fprintf(stderr, "ERROR: recieved incomplete measurement\n");
        return -1;
    } 

    // final check is just is that id even in the map
    if (clonesCAM.at(feat->anchor_cam_id).find(feat->anchor_clone_timestamp) == clonesCAM.at(feat->anchor_cam_id).end()){
        // fprintf(stderr, "ERROR: MEASUREMENT NOT IN CAM VEC\n");
        return -1;
    }

    // Get the position of the anchor pose
    const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

    // Loop through each camera for this feature
    for (auto const &pair : feat->timestamps) {
        // Add CAM_I features
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

            //=====================================================================================
            //=====================================================================================

            // more error handling, as we have some funky behavior on feature inits where timestamps/clone times are misaligned
            if(clonesCAM.at(pair.first).find(feat->timestamps.at(pair.first).at(m)) == clonesCAM.at(pair.first).end()){
                // fprintf(stderr, "FAILED CHECK\n");
                continue;
            }

            // Get the position of this clone in the global
            const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
            const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
            // Convert current position relative to anchor
            Eigen::Matrix<double, 3, 3> R_AtoCi;
            R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
            Eigen::Matrix<double, 3, 1> p_CiinA;
            p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
            Eigen::Matrix<double, 3, 1> p_AinCi;
            p_AinCi.noalias() = -R_AtoCi * p_CiinA;

            //=====================================================================================
            //=====================================================================================

            // Middle variables of the system
            double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
            double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
            double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
            // Calculate residual
            Eigen::Matrix<float, 2, 1> z;
            z << hi1 / hi3, hi2 / hi3;
            Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;
            // Append to our summation variables
            err += pow(res.norm(), 2);
        }
    }

    return err;
}
