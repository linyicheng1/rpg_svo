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
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
  initialize();
}

void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

/**
 * @brief 将图片数据添加到算法中
 *
 * @param img       图片信息
 * @param timestamp 图片时间戳
 */
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  // 1、判断当前状态
  //    没有初始化则初始化
  //    暂停则返回false，直接返回
  // 2、开始计时
  // 3、清空上一次迭代中被丢弃的地图点数据
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  // 清空上一次迭代数据
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  // 新建一帧的信息
  // 1、初始化参数
  // 2、检查图片是否符合要求
  // 3、设置关键点为空
  // 4、创建图像金字塔
  // cam_:相机内参
  // mg.clone():图片信息
  // timestamp:时间戳
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  // @TODO 根据当前系统状态执行不同函数
  // 第一帧、第二帧：初始化部分
  // default:正常执行部分
  // relocalizing:重定位部分
  
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
  {
    std::cout<<"vo state:"<<"STAGE_DEFAULT_FRAME"<<std::endl;
    res = processFrame();
  }
  else if(stage_ == STAGE_SECOND_FRAME)
  {
    std::cout<<"vo state:"<<"STAGE_SECOND_FRAME"<<std::endl;
    res = processSecondFrame();
  }
  else if(stage_ == STAGE_FIRST_FRAME)
  {
    std::cout<<"vo state:"<<"STAGE_FIRST_FRAME"<<std::endl;
    res = processFirstFrame();
  }
  else if(stage_ == STAGE_RELOCALIZING)
  {
    std::cout<<"vo state:"<<"STAGE_RELOCALIZING"<<std::endl;
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));
  }

  // 记录当前帧参数
  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  // 结束一帧的处理
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

FramePtr FrameHandlerMono::lastFrame() { return last_frame_; }
const set<FramePtr>& FrameHandlerMono::coreKeyframes() { return core_kfs_; }
const vector<cv::Point2f>& FrameHandlerMono::initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
const vector<cv::Point2f>& FrameHandlerMono::initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }
DepthFilter* FrameHandlerMono::depthFilter() const { return depth_filter_; }

/**
 * @brief  初始化：处理第一帧数据
 * @return 处理结果
 */
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  // 第一帧的位置设置为原点
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  // 光流初始化，添加第一帧数据
  // 1.设置第一帧为参考帧
  // 2.提取特征点
  // 3.特征点少于100，丢弃
  // new_frame_:当前帧图片信息
  // return:是否满足第一帧要求
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  // 第一帧一定是一个关键帧
  // 1、设置关键帧标志位为1
  // 2、寻找中间+四周共5个点作为当前帧的特征点
  new_frame_->setKeyframe();
  // 添加到地图中
  map_.addKeyframe(new_frame_);
  // 切换状态为第二帧处理
  stage_ = STAGE_SECOND_FRAME;
  // 返回处理结果为关键帧
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

/**
 * @brief  初始化：处理第二帧数据
 * @return 处理结果
 */
FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  // 光流初始化的第二帧
  // @TODO
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  // 结果为RESULT_FAILURE，则重新开始（包括第一帧）
  // 结果为RESULT_NO_KEYFRAME,则重新找第二帧（第一帧不变）
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
  // 使用BA优化二者的位置关系
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif
  // 第二帧也设置成为关键帧
  // 1、设置关键帧标志位为1
  // 2、寻找中间+四周共5个点作为当前帧的特征点
  new_frame_->setKeyframe();

  double depth_mean, depth_min;
  // 计算两帧构成地图的深度均值、极小值
  // new_frame_：当前帧；depth_mean:深度平均值；depth_min:深度最小值；
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  // 添加到深度滤波器中
  //
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  // 添加到地图中
  map_.addKeyframe(new_frame_);
  // 切换当前状态到正常运行状态
  stage_ = STAGE_DEFAULT_FRAME;
  // 此时初始化结束，清空一些初始化数据，因为在丢失后可能要重新初始化
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

/**
 * @brief  数据正常处理流程
 * @return 处理结果
 */
FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  // Set initial pose TODO use prior
  // 设置新的一帧位置和上一帧的保持一致
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  /***********************采用直接法直接计算位姿**********************/
  SVO_START_TIMER("sparse_img_align");
  // 图像对齐类初始化
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, true, false);
                          
  // 进行对齐，获取初始值
  // ref_frame:参考帧 cur_frame:当前帧
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);
  std::cout<<"Image Align: Tracked = "<<img_align_n_tracked<<std::endl;

  /***********************重新寻找最优匹配点位置**********************/
  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  // 重新寻找更优的特征点对应位置
  // new_frame_ 当前帧
  // overlap_kfs_ 之前所有的关键帧
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  // 重新计算出来的匹配点
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  std::cout<<"Reprojection:"<<repr_n_mps<<"Matches = "<<repr_n_new_references<<std::endl;
  // 重新投影得到的匹配点不够的话就不能继续后面的优化了
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }

  /***********************使用BA对位姿进行优化**********************/
  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  /***********************关键帧的选择**********************/
  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  // if(tracking_quality_ == TRACKING_INSUFFICIENT)
  // {
  //   new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
  //   return RESULT_FAILURE;
  // }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();
  std::cout<<"New keyframe selected."<<std::endl;
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif
  /***********************把数据传递给深度滤波器**********************/
  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  // 添加新的一帧到地图中
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
