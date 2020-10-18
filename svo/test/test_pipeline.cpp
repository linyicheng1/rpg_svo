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

#include <time.h>
#include <sstream>
#include <mmintrin.h>  //MMX
#include <xmmintrin.h> //SSE(include mmintrin.h)
#include <emmintrin.h> //SSE2(include xmmintrin.h)
#include <pmmintrin.h> //SSE3(include emmintrin.h)
#include <tmmintrin.h> //SSSE3(include pmmintrin.h)
#include <smmintrin.h> //SSE4.1(include tmmintrin.h)
#include <nmmintrin.h> //SSE4.2(include smmintrin.h)
#include <wmmintrin.h> //AES(include nmmintrin.h)
#include <immintrin.h> //AVX(include wmmintrin.h)

namespace svo
{
  // 算法流程类
  class BenchmarkNode
  {
    // 相机模型类
    vk::AbstractCamera *cam_;
    // vo算法类
    svo::FrameHandlerMono *vo_;

  public:
    BenchmarkNode();
    ~BenchmarkNode();
    void runFromFolder();
    void loadImages(const std::string &strImagePath, const std::string &strPathTimes,
                    std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps);
    cv::Mat preProcess(cv::Mat raw);
    std::vector<int> expose_time; 
    int num = 0;
  };

  /**
 * @brief 构造函数
 *        初始化相机模型、vo算法类
 */
  BenchmarkNode::BenchmarkNode()
  {
    //初始化相机模型
    // 图片大小：752 x 480 焦距 fx：315.5 fy：315.5 图像中心：cx：376.0 cy：240.0
    // 原始相机参数
    // cam_ = new vk::PinholeCamera(752, 480, 315.5, 315.5, 376.0, 240.0);
    // 修改后的相机参数
    cam_ = new vk::PinholeCamera(752, 480, 605.7455777299853, 607.9735414995575, 370.360573680134,206.06348100221066);
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
     * @brief 计算梯度
     * */
  cv::Mat BenchmarkNode::preProcess(cv::Mat raw)
  {
    int w = 752;
    int h = 480;
    // int size = (w - 2) * (h - 2); // 边缘一个像素不计算梯度
    // __m128i com1, com2, gx, gy, com11, com22, gx_1, gy_1, tmp;
    cv::Mat dst = raw.clone();
    double k = (double)expose_time[0] / (double)expose_time[num] * 0.1;
    raw.convertTo(dst, -1, k, 0);
    
    // for (int i = 0; i < h - 1; i++)
    // {
    //   for (int j = 0; j < w - 1; j += 16)
    //   {
    //     dst.data[i * w + j] = (unsigned char )((float)raw.data[i * w + j] * k);
    //     // const unsigned char *A = (unsigned char *)(raw.data + i * w + j);
    //     // const unsigned char *B = (unsigned char *)(raw.data + i * w + j + 1);
    //     // const unsigned char *C = (unsigned char *)(raw.data + (i + 1) * w + j);
    //     // const unsigned char *D = (unsigned char *)(raw.data + (i + 1) * w + j + 1);
    //     // const __m128i pSmooth_A = _mm_loadu_si128((const __m128i *)(A));
    //     // const __m128i pSmooth_B = _mm_loadu_si128((const __m128i *)(B));
    //     // const __m128i pSmooth_C = _mm_loadu_si128((const __m128i *)(C));
    //     // const __m128i pSmooth_D = _mm_loadu_si128((const __m128i *)(D));
    //     // __m128i *grad = (__m128i *)(dst.data + i * w + j);


    //     // gx = _mm_subs_epu8(pSmooth_A, pSmooth_B);
    //     // gy = _mm_subs_epu8(pSmooth_C, pSmooth_A);
    //     // gx_1 = _mm_subs_epu8(pSmooth_B, pSmooth_A);
    //     // gy_1 = _mm_subs_epu8(pSmooth_A, pSmooth_C);

    //     // gx = _mm_adds_epu8(gx, gx_1);
    //     // gy = _mm_adds_epu8(gy, gy_1);
    //     // *grad = _mm_adds_epu8(gx, gy);
    //   }
    // }
    return dst;
  }

  /**
 * @brief 从文件中读取图片信息，输入算法中进行计算
 */
  void BenchmarkNode::runFromFolder()
  {

    /**************** 对EURoC数据集支持 ********************/
    std::string dir = "/home/lyc/slam/dataSet/MH_01_easy/mav0/cam0/";
    
    cv::Mat mat;
    std::vector<std::string> strImages;
    std::vector<double> timeStamps;
    // data.csv
    std::string data_dir = dir + "data.csv";
    std::string pic_dir = dir + "data";
    loadImages(pic_dir, data_dir, strImages, timeStamps);
    clock_t startTime, endTime;
    while (true)
    {
      std::string file = strImages.at(num);
      mat = cv::imread(file, cv::IMREAD_GRAYSCALE);
      if (mat.empty())
      {
        break;
      }
      //cv::Mat pre = preProcess(mat);
      vo_->addImage(mat, timeStamps[num]);
      if (vo_->lastFrame() != NULL)
      {
        std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                  << "#Features: " << vo_->lastNumObservations() << " \t"
                  << "Proc. Time: " << vo_->lastProcessingTime() * 1000 << "ms \n";
        // access the pose of the camera via vo_->lastFrame()->T_f_w_.
      }
      cv::imshow("test", mat);
      //cv::imshow("pre", pre);

      cv::waitKey(1);
      num++;
    }

    /**************** 原始的读取图片运行函数 ****************/
    /*
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
  */
  }
  void BenchmarkNode::loadImages(const std::string &strImagePath, const std::string &strPathTimes,
                                 std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps)
  {
    std::ifstream fTimes;
    std::string s;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    std::getline(fTimes, s);
    while (!fTimes.eof())
    {
      std::getline(fTimes, s);
      if (!s.empty())
      {
        std::stringstream ss(s);
        std::string time;
        std::getline(ss, time, ',');
        vstrImages.push_back(strImagePath + "/" + time + ".png");
        double t;
        std::stringstream(time) >> t;
        vTimeStamps.push_back(t / 1e9);
        // std::getline(ss, time, ',');
        // std::getline(ss, time, ',');
        // expose_time.push_back(std::stoi(time));
      }
    }
  }
} // namespace svo
//测试算法流程
int main(int argc, char **argv)
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
