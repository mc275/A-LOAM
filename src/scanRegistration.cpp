// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


/**
* imu为 x轴向前,y轴向左,z轴向上的右手坐标系
* velodyne lidar被安装为 x轴向前,y轴向左,z轴向上的右手坐标系
* LOAM 论文坐标系 z轴向前,x轴向左,y轴向上的右手坐标系
* 原有LOAM代码会使用laserCloudHandler将坐标进行交换，A LOAM没有这么做
**/


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

/// 扫描周期, velodyne频率10Hz，周期0.1s
const double scanPeriod = 0.1;

///初始化控制变量
const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;

/// 激光雷达线数
int N_SCANS = 64;

///点云曲率, 40000为一帧点云中点的最大数量
float cloudCurvature[400000];

/// 曲率点对应的序号
int cloudSortInd[400000];

/// 点是否经过筛选的标志：0-未筛选过，1-筛选过
int cloudNeighborPicked[400000];

/// 点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
int cloudLabel[400000];


/**
 * @brief 点云曲率比较谓词函数
 *
 * @param i 索引1
 * @param j 索引2
 * @return i与j对应的曲率
    * @retval true cloudCurvature[i]<cloudCurvature[j]
    * @retval false cloudCurvature[i]>cloudCurvature[j]
 */
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

/// 点云publisher对象
ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;

/// 激光点云线束publisher对象
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;                     ///< 每条线输出

double MINIMUM_RANGE = 0.1;                     ///< 测距量程下限


/**
 * @brief 删除测量距离低于阈值thres的点云
 *
 * @param cloud_in  输入点云
 * @param cloud_out 输出点云
 * @param thres 测量阈值下限
 * @return 无
    * @retval
 */
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 输入和输出不相同的点云，浅拷贝数据
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}




/**
 * @brief 当前node的算法主函数, 视线点云预处理、特征提取和发送
 *
 * @param laserCloudMsg 当前帧激光点云
 * @return 无
    * @retval
 */
// 接收点云数据，velodyne雷达坐标系安装为x轴向前，y轴向左，z轴向上的右手坐标系
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 系统状态判定，可丢弃一定数量的点云数据
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;                             // 定时器
    TicToc t_prepare;                           // 定时器

    //记录每个scan(线数)有曲率的点的开始和结束索引
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    /**STEP1 读取点云数据并剔除错误数据*************/
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;                       // 旧点云到新点云的索引

    // 删除超出量程范围的点云
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    /**STEP2 筛选满足条件的点云*************/
    /**Step2.1 计算激光点云横向视场角*/
    int cloudSize = laserCloudIn.points.size();
    //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    //lidar scan结束点的旋转角,加2*pi使点云旋转周期为2*pi
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    //正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正,允许不是一个圆周扫描
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    /**Step2.2 筛选满足视角条件的点云，计算点云观测时间*/
    bool halfPassed = false;                                                //扫描线是否旋转过半
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);      // 不同激光线束点云分开保存
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // 计算激光点仰角
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI; //激光点云角度值
        int scanID = 0;

        // 筛选雷达点云俯仰角满足阈值的点云，剔除外点
        if (N_SCANS == 16)
        {
            // 计算点云所在的16激光雷达线编号,四舍五入，[-15,15] 对应[0,29]
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }

        else if (N_SCANS == 64)
        {
            // 计算点云位于的64线激光雷达线编号，四舍五入
            // 64线雷达，仰角不是均分
            // +2 ～ -8.33：  1/3 degree
            //  -8.83 ～ -24.33： 1/2 degree
            if (angle >= -8.83)
            {
                scanID = int((2 - angle) * 3.0 + 0.5);
            }
            else
            {
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
            }
            // 只使用[0,50]线范围内的点云
            // use [0 50]  > 50 remove outlies
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);       // 点云旋转角

        //根据扫描线是否旋转过半，选择与起始位置还是终止位置进行差值计算
        if (!halfPassed)
        {
            //确保-pi/2 < ori - startOri < 3*pi/2
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }


            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            //确保 -pi/2 < endOri - ori < pi*3/2
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        // -0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
        float relTime = (ori - startOri) / (endOri - startOri);     // 计算点云接收时间
        // 点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）
        // 匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
        point.intensity = scanID + scanPeriod * relTime;
        laserCloudScans[scanID].push_back(point); 
    }

    //获得有效范围内的点的数量
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    /**STEP3 点云特征提取*************/
    /**Step3.1 点云合并*/
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());   //
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;       // 当前scan(线) i有曲率的首个点索引
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;         // 当前scan(线) i有曲率的最后点索引
    }

    printf("prepare time %f \n", t_prepare.toc());              // 打印计算耗时
    /**Step3.2 点云曲率计算*/
    //使用每个点的前后五个点计算曲率，因此前五个与最后五个点跳过
    for (int i = 5; i < cloudSize - 5; i++)
    {
        // 论文中的曲率计算公式，当前只计算了分子
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x
                      + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x
                      - 10 * laserCloud->points[i].x
                      + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x
                      + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y
                      + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y
                      - 10 * laserCloud->points[i].y
                      + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y
                      + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z
                      + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z
                      - 10 * laserCloud->points[i].z
                      + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z
                      + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }

    // Todo: 缺少点云的进一步筛选，论文中提到两类点云应剔除

    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;       // 边点特征
    pcl::PointCloud<PointType> cornerPointsLessSharp;   // 边点候选特征
    pcl::PointCloud<PointType> surfPointsFlat;          // 面点特征
    pcl::PointCloud<PointType> surfPointsLessFlat;      // 剩余点云

    /**Step3.3 点云特征筛选*/
    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        // 当前scan中没有曲率点云
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;


        // 将每个scan的曲率点云6等分,确保周围都有点被选作特征点
        // 等分点云包括2个边点特征和4个面点特征
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {

            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;           // 6等分点云中每份的起点
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1; // 6等分点云中每份的终点

            // 每段点云索引按照曲率升序排列
            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            // 筛选大曲率点云作为边点特征候选
            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; 

                // 针对未筛选过的点云选择大曲率点云
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    //挑选曲率最大的前2个点放入sharp点集合
                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;    //曲率大小标志位
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //挑选曲率最大的前20个点放入less sharp点集合
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    //置位筛选标志
                    cloudNeighborPicked[ind] = 1; 


                    // 为了特征分布均匀，大曲率点周围的一般曲率点不作为特征候选(论文中描述)
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 筛选小曲率点云作为边点特征候选
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1; //曲率大小标志位
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    //只选最小的四个，剩下的Label==0,就都是曲率比较小的
                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    // 特征均匀分布化
                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            //将剩余的点（包括之前被排除的点）全部归入平面点中less flat类别中
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        // 体素滤波降采样less flat，降低点数
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());


    /*********STEP4 发送点云*************/
    // publish满足条件的所有的点
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    // publish 边点
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    // publish 边点候选特征
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    // publish 面点特征
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    // publish 其余点云
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}


/**
 * @brief 当前节点主函数
 *
 * @param argc 输入参数个数
 * @param argv 输入参数字符串头指针
 * @return 0
    * @retval
 */
int main(int argc, char **argv)
{
    /**STEP1 初始化ros和节点*************/
    /** Step1.1 初始化ROS，指定节点名称*/
    // argc和argv解析 rosrun package_name node_name param_name:=param_value 中
    // remapping arguments param:=param_value
    ros::init(argc, argv, "scanRegistration");

    /** Step1.2 实例化节点创建节点句柄，初始化节点，是当前节点程序和系统交互的主要机制*/
    ros::NodeHandle nh;

    /**Step1.3 设置节点参数*/
    nh.param<int>("scan_line", N_SCANS, 16);                    // 激光雷达线数，N_SCANS, 默认为16
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);      // 最近探测距离，MINIMUM_RANGE， 默认为0.1

    printf("scan line number %d \n", N_SCANS);

    // 当前仅支持16线和64线
    if(N_SCANS != 16 && N_SCANS != 64)
    {
        printf("only support velodyne with 16 or 64 scan line!");
        return 0;
    }

    /**STEP2 初始化Subscriber和Publisher*************/
    /** Step2.1 定义Subscriber，订阅话题名称为/velodyne_points，消息类型为sensor_msgs::PointCloud2
    * 缓存buffer大小100， 订阅节点回调函数 laserCloudHandler */
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    /**Step2.2 定义Publisher*/
    // 发布名称为/velodyne_cloud_2 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    // 发布名称为/laser_cloud_sharp 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    // 发布名称为/laser_cloud_less_sharp 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    // 发布名称为/laser_cloud_flat 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    // 发布名称为/laser_cloud_less_flat 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    // 发布名称为/laser_remove_points 话题，消息类型为sensor_msgs::PointCloud2， 缓存buffer大小100
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    // 发送激光每条线
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }

    /**STEP3 循环处理Subscriber的回调函数*************/
    // 循环监听回调队列，处理回调函数，直到ros::ok()返回false
    // ros::ok() 返回false的几种情况：
    //  SIGINT收到(Ctrl-C)信号
    //  另一个同名节点启动，会先中止之前的同名节点
    //  ros::shutdown()被调用
    //  所有的ros::NodeHandles被销毁
    ros::spin();

    return 0;
}
