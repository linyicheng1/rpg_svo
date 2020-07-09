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
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include "test_utils.h"

namespace svo {
// 算法流程类
class BenchmarkNode
{
  // 相机模型类
  vk::AbstractCamera* cam_;
  // vo算法类
  svo::FrameHandlerMono* vo_;

public:
  BenchmarkNode();
  ~BenchmarkNode();
  void runFromFolder();
};

/**
 * @brief 构造函数
 *        初始化相机模型、vo算法类
 */
BenchmarkNode::BenchmarkNode()
{
  //初始化相机模型
  // 图片大小：752 x 480 焦距 fx：315.5 fy：315.5 图像中心：cx：376.0 cy：240.0
  cam_ = new vk::PinholeCamera(752, 480, 315.5, 315.5, 376.0, 240.0);
  // @TODO 初始化vo算法，内部新建深度滤波器线程
  vo_ = new svo::FrameHandlerMono(cam_);
  // 设置开始标志位为1
  vo_->start();
}

BenchmarkNode::~BenchmarkNode()
{
  delete vo_;
  delete cam_;
}

/**
 * @brief 从文件中读取图片信息，输入算法中进行计算
 */
void BenchmarkNode::runFromFolder()
{
  for(int img_id = 2; img_id < 188; ++img_id)
  {// 遍历188张图片

    // load image
    // 获取文件名称
    std::stringstream ss;
    ss << svo::test_utils::getDatasetDir() << "/sin2_tex2_h1_v8_d/img/frame_"
       << std::setw( 6 ) << std::setfill( '0' ) << img_id << "_0.png";
    if(img_id == 2)
      std::cout << "reading image " << ss.str() << std::endl;
    // 根据文件名称读取图片
    cv::Mat img(cv::imread(ss.str().c_str(), 0));
    assert(!img.empty());

    // process frame
    // @TODO 将图片添加到算法中，进行处理计算
    vo_->addImage(img, 0.01*img_id);

    // display tracking quality
    // 输出调试信息
    if(vo_->lastFrame() != NULL)
    {
    	std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                  << "#Features: " << vo_->lastNumObservations() << " \t"
                  << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \n";

    	// access the pose of the camera via vo_->lastFrame()->T_f_w_.
    }
  }
}

} // namespace svo
//测试算法流程
int main(int argc, char** argv)
{
  {
    // 初始化类
    svo::BenchmarkNode benchmark;
    // 从文件中运行
    benchmark.runFromFolder();
  }
  printf("BenchmarkNode finished.\n");
  return 0;
}

