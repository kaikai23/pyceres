#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <sophus/sim3.hpp>

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;

/* Adapted from:
 * https://github.com/ceres-solver/ceres-solver/blob/master/examples/slam/pose_graph_3d/pose_graph_3d_error_term.h
 * https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/autodiff_benchmarks/relative_pose_error.h
 * Use right-hand multiplication like in GTSAM.
 * The covariance is expressed in the frame j.
 */
struct PoseGraphRelativeCost {
 public:
  explicit PoseGraphRelativeCost(const Eigen::Vector4d& qvec_j_i,
                                 const Eigen::Vector3d& tvec_j_i,
                                 const Matrix6d covariance)
      : meas_q_j_i_(qvec_j_i),
        meas_t_j_i_(tvec_j_i),
        sqrt_information_(covariance.inverse().llt().matrixL()) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec_j_i,
                                     const Eigen::Vector3d& tvec_j_i,
                                     const Matrix6d covariance) {
    return (new ceres::AutoDiffCostFunction<PoseGraphRelativeCost, 6, 4, 3, 4, 3>(
        new PoseGraphRelativeCost(qvec_j_i, tvec_j_i, covariance)));
  }

  template <typename T>
  bool operator()(const T* const qvec_i_w, const T* const tvec_i_w,
                  const T* const qvec_j_w, const T* const tvec_j_w,
                  T* residuals_ptr) const {
    T qvec_i_j[4];
    const T qvec_w_j[4] = {qvec_j_w[0], -qvec_j_w[1], -qvec_j_w[2], -qvec_j_w[3]};
    ceres::QuaternionProduct(qvec_i_w, qvec_w_j, qvec_i_j);

    Eigen::Matrix<T, 3, 1> t_i_j;
    ceres::UnitQuaternionRotatePoint(qvec_i_j, tvec_j_w, t_i_j.data());
    t_i_j = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(tvec_i_w) - t_i_j;

    T qvec_res[4];
    const Eigen::Matrix<T, 4, 1> meas_q_j_i = meas_q_j_i_.cast<T>();
    ceres::QuaternionProduct(meas_q_j_i.data(), qvec_i_j, qvec_res);
    ceres::QuaternionToAngleAxis(qvec_res, residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> t_res(residuals_ptr + 3);
    ceres::UnitQuaternionRotatePoint(meas_q_j_i.data(), t_i_j.data(), t_res.data());
    t_res += meas_t_j_i_.cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Measurement of relative pose from i to j.
  Eigen::Vector4d meas_q_j_i_;
  Eigen::Vector3d meas_t_j_i_;
  const Matrix6d sqrt_information_;
};

struct PoseGraphAbsoluteCost {
 public:
  explicit PoseGraphAbsoluteCost(const Eigen::Vector4d& qvec_i_w,
                                 const Eigen::Vector3d& tvec_i_w,
                                 const Matrix6d covariance)
      : q_i_w_(qvec_i_w),
        t_i_w_(tvec_i_w),
        sqrt_information_(covariance.inverse().llt().matrixL()) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec_i_w,
                                     const Eigen::Vector3d& tvec_i_w,
                                     const Matrix6d covariance) {
    return (new ceres::AutoDiffCostFunction<PoseGraphAbsoluteCost, 6, 4, 3>(
        new PoseGraphAbsoluteCost(qvec_i_w, tvec_i_w, covariance)));
  }

  template <typename T>
  bool operator()(const T* const qvec_i_w, const T* const tvec_i_w,
                  T* residuals_ptr) const {
    T qvec_res[4];
    const T meas_q_i_w[4] = {T(q_i_w_(0)), T(q_i_w_(1)), T(q_i_w_(2)), T(q_i_w_(3))};
    const T qvec_w_i[4] = {qvec_i_w[0], -qvec_i_w[1], -qvec_i_w[2], -qvec_i_w[3]};
    ceres::QuaternionProduct(meas_q_i_w, qvec_w_i, qvec_res);
    ceres::QuaternionToAngleAxis(qvec_res, residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> t_res(residuals_ptr + 3);
    ceres::UnitQuaternionRotatePoint(qvec_res, tvec_i_w, t_res.data());
    t_res = t_i_w_.cast<T>() - t_res;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Measurement of absolute pose from world to i.
  Eigen::Vector4d q_i_w_;
  Eigen::Vector3d t_i_w_;
  const Matrix6d sqrt_information_;
};


struct PoseGraphRelativeSim3Cost {
 public:
  explicit PoseGraphRelativeSim3Cost(const double s_j_i,
                                 const Eigen::Vector4d& qvec_j_i,
                                 const Eigen::Vector3d& tvec_j_i,
                                 const Matrix6d covariance)
      : sqrt_information_(covariance.inverse().llt().matrixL()) {
    double* p_j_i;
    ceres::MatrixAdapter<double, 3, 1> R_j_i(p_j_i);
    ceres::QuaternionToRotation(qvec_j_i.data(), R_j_i);
    Eigen::Matrix4d T_j_i;
    T_j_i << s_j_i * R_j_i(0,0), s_j_i * R_j_i(0,1), s_j_i * R_j_i(0,2), tvec_j_i[0],
             s_j_i * R_j_i(1,0), s_j_i * R_j_i(1,1), s_j_i * R_j_i(1,2), tvec_j_i[1],
             s_j_i * R_j_i(2,0), s_j_i * R_j_i(2,1), s_j_i * R_j_i(2,2), tvec_j_i[2],
             0,                  0,                  0,                  1;
    Sophus::Sim3d mear_Sim_j_i_(T_j_i);
  }

  static ceres::CostFunction* Create(const double s_j_i,
                                     const Eigen::Vector4d& qvec_j_i,
                                     const Eigen::Vector3d& tvec_j_i,
                                     const Matrix6d covariance) {
    return (new ceres::AutoDiffCostFunction<PoseGraphRelativeSim3Cost, 7, 1, 4, 3, 1, 4, 3>(
        new PoseGraphRelativeSim3Cost(s_j_i, qvec_j_i, tvec_j_i, covariance)));
  }

  template <typename T>
  bool operator()(const T* s_i_w, const T* const qvec_i_w, const T* const tvec_i_w,
                  const T* s_j_w, const T* const qvec_j_w, const T* const tvec_j_w,
                  T* residuals_ptr) const {
    double* p_i_w;
    ceres::MatrixAdapter<double, 3, 1> R_i_w(p_i_w);
    ceres::QuaternionToRotation(qvec_i_w, R_i_w);
    Eigen::Matrix<T, 4, 4> T_i_w;
    T_i_w << s_i_w[0] * R_i_w(0,0), s_i_w[0] * R_i_w(0,1), s_i_w[0] * R_i_w(0,2), tvec_i_w[0],
             s_i_w[0] * R_i_w(1,0), s_i_w[0] * R_i_w(1,1), s_i_w[0] * R_i_w(1,2), tvec_i_w[1],
             s_i_w[0] * R_i_w(2,0), s_i_w[0] * R_i_w(2,1), s_i_w[0] * R_i_w(2,2), tvec_i_w[2],
             0,                  0,                  0,                  1;
    Sophus::Sim3<T> Sim_i_w(T_i_w);

    double* p_j_w;
    ceres::MatrixAdapter<double, 3, 1> R_j_w(p_j_w);
    ceres::QuaternionToRotation(qvec_j_w, R_j_w);
    Eigen::Matrix<T, 4, 4> T_j_w;
    T_j_w << s_j_w[0] * R_j_w(0,0), s_j_w[0] * R_j_w(0,1), s_j_w[0] * R_j_w(0,2), tvec_j_w[0],
             s_j_w[0] * R_j_w(1,0), s_j_w[0] * R_j_w(1,1), s_j_w[0] * R_j_w(1,2), tvec_j_w[1],
             s_j_w[0] * R_j_w(2,0), s_j_w[0] * R_j_w(2,1), s_j_w[0] * R_j_w(2,2), tvec_j_w[2],
             0,                  0,                  0,                  1;
    Sophus::Sim3<T> Sim_j_w(T_j_w);
    Eigen::Map<Eigen::Matrix<T, 7, 1>> residuals(residuals_ptr);
    residuals = (meas_Sim_j_i_ * Sim_i_w * Sim_j_w.inverse());
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Measurement of relative pose from i to j.
  Sophus::Sim3d meas_Sim_j_i_;
  const Matrix7d sqrt_information_;
};
