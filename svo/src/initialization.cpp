// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

/**
 * @brief 光流初始化，添加第一帧的数据
 *        1.设置第一帧为参考帧
 *        2.提取特征点
 *        3.特征点少于100，丢弃
 * @param frame_ref 当前帧信息
 * @return 当前帧是否符合初始化要求
 */
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  // 重置特征点和参考帧数据
  reset();
  // 检测特征点
  // frame_ref：当前帧
  // px_ref_:特征点位置
  // f_ref_:特征点的方向向量（没有计算）
  detectFeatures(frame_ref, px_ref_, f_ref_);
  // 如果特征点太少了，就不要
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  // 特征点足够，设置当前帧为参考帧
  frame_ref_ = frame_ref;
  // 将所有特征点保存下来
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

/**
 * @brief 光流初始化，第二帧数据处理
 * @param frame_cur 当前帧信息
 * @return 处理结果
 */
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  // 使用光流跟踪
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");
  // 如果跟踪的点太少则初始化失败
  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;
  // 计算平均视差
  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  // 如果均值视差太小了，就等下一帧作为第二帧，这样视差就会大一些
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;
  // 计算单应矩阵
  // f_ref_ 参考帧的球坐标
  // f_cur_ 当前帧的球坐标
  // focal_length 焦距
  // 阈值
  // inliers_        返回内点
  // xyz_in_cur_     当前坐标系下
  // T_cur_from_ref_ 和参考帧之间的坐标变换关系
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");
  // RANSAC算法内点数太少的话，也需要重新初始化
  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  // 每一个地图点的z坐标放入depth_vec中
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  // 获取深度的平均值
  double scene_depth_median = vk::getMedian(depth_vec);
  // 计算缩放系数
  double scale = Config::mapScale()/scene_depth_median;
  // 计算当前帧相对于世界坐标系的位姿
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  // 对于每一个内点都创建一个3d点
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  // 遍历所有的内点
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    // 1、在参考帧视野范围内 2、在摄像头前方
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      // 计算特征点在世界坐标系下的位置
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      // 构造一个3d点
      Point* new_point = new Point(pos);
      // 构造当前帧的特征点
      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      // 将特征点加入当前帧
      frame_cur->addFeature(ftr_cur);
      // 3d点内记录对应特征点
      new_point->addFrameRef(ftr_cur);
      // 同样加入到参考帧中
      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}

/**
 * @brief 重置当前特征点和参考帧，在第一帧时调用和结束初始化时
 */
void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

/**
 * @brief 检测一帧的特征点
 * @param frame  当前帧数据
 * @param px_vec 特征点位置vector
 * @param f_vec  特征点方向向量vector
 */
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  // FastDetector类提取特征点
  // 初始化赋值
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  // 检测特征点,即为得分大于一定值的角点
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // 清空之前的，按现在大小分配内存
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  // 记录需要的特征点位置信息
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}

/**
 * @brief 使用光流去跟踪
 * @param frame_ref     参考帧
 * @param frame_cur     当前帧
 * @param px_ref        参考特征点位置
 * @param px_cur        当前帧特征点位置
 * @param f_ref         参考帧特征点单位球坐标
 * @param f_cur         当前帧特征点单位球坐标
 * @param disparities   每一对匹配的视差
 */
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);

  // 调用cv函数，计算光流
  /**
   * @param prevImg 参考帧图片
   * @param nextImg 当前帧图片
   * @param px_ref  参考帧特征点
   * @param px_cur  当前帧特征点
   * @param status  记录是否搜索到了相似点
   * @param error   错误码
   * @param winSize 搜索窗口大小
   * @param maxLevel 金字塔等级
   */
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // 计算方向向量和视差
  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());

  // 遍历所有参考帧的特征点
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    // 如果当前特征点没有在窗口内搜索到相似点
    if(!status[i])
    {
      // 从列表中移除
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    // 存在匹配点，则计算视差和单位球坐标
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    // 视差即计算两点坐标差值
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    // 下一次迭代
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

/**
 * @brief 通过计算单应矩阵，恢复相机相对位姿和特征点3d坐标
 * @param f_ref                   参考帧特征点球坐标
 * @param f_cur                   当前帧特征点球坐标
 * @param focal_length            焦距
 * @param reprojection_threshold  重投影阈值
 * @param inliers                 返回内点
 * @param xyz_in_cur              当前帧坐标系下特征点xyz坐标
 * @param T_cur_from_ref          和参考帧相对的坐标变换关系
 */
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d, aligned_allocator<Vector2d> > uv_ref(f_ref.size());
  vector<Vector2d, aligned_allocator<Vector2d> > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }
  // 调库实现对单应矩阵计算及求解相对位姿
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
