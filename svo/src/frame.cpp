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

#include <stdexcept>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
#include <fast/fast.h>

namespace svo {

int Frame::frame_counter_ = 0;

/**
 * @brief 帧类的构造函数
 * @param cam       相机内参
 * @param img       图像信息
 * @param timestamp 时间戳
 */
Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
    id_(frame_counter_++),
    timestamp_(timestamp),
    cam_(cam),
    key_pts_(5),
    is_keyframe_(false),
    v_kf_(NULL)
{
  // 调用初始化函数
  // 1、检查图片是否符合要求
  // 2、设置关键点为空
  // 3、创建图像金字塔
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* i){delete i;});
}

/**
 * @brief 帧类初始化函数
 *        1、检查图片是否符合要求
 *        2、设置关键点为空
 *        3、创建图像金字塔
 * @param img 图像信息
 */
void Frame::initFrame(const cv::Mat& img)
{
  // check image
  // 检查图片 1. 空 2.编码格式 3.长宽要求
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to NULL
  // 设置所有关键点为空
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

  // Build Image Pyramid
  // 创建图像金字塔
  // img:原始图像信息
  // n_levels:金字塔层数
  // 金字塔图片vector
  frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
}

/**
 * @brief 设置为关键帧
 */
void Frame::setKeyframe()
{
  // 关键帧标志位置1
  is_keyframe_ = true;
  // 设置关键点
  // 特征点一共5个，中间+四周
  setKeyPoints();
}

void Frame::addFeature(Feature* ftr)
{
  fts_.push_back(ftr);
}

/**
 * @brief 设置关键点，在设置关键帧时调用
 *          特征点一共5个，中间+四周
 *        遍历所有特征点，挑选满足条件的点作为关键点
 */
void Frame::setKeyPoints()
{
  // 将所有关键点数据清空
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;
  // 遍历所有特征点
  // 调用函数 checkKeyPoints(ftr) 判断是否满足关键点条件
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });
}

/**
 * @brief 判断特征点是否满足关键点条件
 *        特征点一共5个，中间+四周
 * @param ftr 特征点
 */
void Frame::checkKeyPoints(Feature* ftr)
{
  // 获取相机中心位置
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;

  // center pixel
  // 找到最靠近中间的点
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;
  // 找到最右下角的点
  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }
  // 找到最右上角的点
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
      key_pts_[2] = ftr;
  }
  // 找到最左上角的点
  if(ftr->px[0] < cv && ftr->px[1] < cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }
  // 找到最左下角的点
  if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[4]->px[0]-cu) * (key_pts_[4]->px[1]-cv))
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(Feature* ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
    if(i == ftr) {
      i = NULL;
      found = true;
    }
  });
  if(found)
    setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w;
  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);
  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}


/// Utility functions for the Frame class
namespace frame_utils {

/**
 * @brief 创建图像金字塔
 * @param img_level_0 原始图像信息
 * @param n_levels    金字塔层数
 * @param pyr         图像金字塔vector
 */
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  // 设置vector大小
  pyr.resize(n_levels);
  // 第一层就是原始图像
  pyr[0] = img_level_0;
  for(int i=1; i<n_levels; ++i)
  {
    // 每一层都是上一层的降采样，变成原来图片的1/2
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}

bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.fts_.size());
  depth_min = std::numeric_limits<double>::max();
  for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
  {
    if((*it)->point != NULL)
    {
      const double z = frame.w2f((*it)->point->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min);
    }
  }
  if(depth_vec.empty())
  {
    SVO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
    return false;
  }
  depth_mean = vk::getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace svo
