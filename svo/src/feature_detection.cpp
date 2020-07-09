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

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

/**
 * @brief 快速检测，初始化参数
 * @param img_width    图像宽度
 * @param img_height   图像高度
 * @param cell_size    cell大小
 * @param n_pyr_levels 金字塔层数
 */
FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

/**
 * @brief 特征点检测函数
 * @param frame                当前帧
 * @param img_pyr              图像金字塔
 * @param detection_threshold  阈值
 * @param fts                  得到的特征点
 */
void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
  // 构造角点结构体vector,设置最大角点数量 grid_n_cols_*grid_n_rows_
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  // 遍历所有金字塔图像
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    // 计算当前金字塔层级的图像缩放系数
    const int scale = (1<<L);// scale = 2^{L}
    vector<fast::fast_xy> fast_corners;//存储角点vector
#if __SSE2__
    // 调用FAST库函数，进行角点检测
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif

    vector<int> scores, nm_corners;
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])
        continue;
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }

  // Create feature for every corner that has high enough corner score
  // 对于有足够高的得分的角点构造成特征点
  std::for_each(corners.begin(), corners.end(), [&](Corner& c)
  {// 遍历所有的角点
    // 要求得分大于阈值
    if(c.score > detection_threshold)
    {
        // 构造特征点
        // 1、所属帧 2、特征点位置 3、金字塔层数
        fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
    }
  });

  resetGrid();
}

} // namespace feature_detection
} // namespace svo

