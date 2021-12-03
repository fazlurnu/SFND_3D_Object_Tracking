
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<double> DistanceEuclideanCoord;
    
    // iterating over matching points in order to create a vector with all distances
    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    {
        if(boundingBox.roi.contains(kptsCurr.at(it1->trainIdx).pt)) // checking if the keypoint is inside of the bounding box of the current frame.
        { 
            // calculating euclidean Distance with opencv function for each matched points
            DistanceEuclideanCoord.push_back(cv::norm(kptsCurr.at(it1->trainIdx).pt - kptsPrev.at(it1->queryIdx).pt)); 
        }
    }



    // calculating mean and standar deviation.
    double sum = accumulate(begin(DistanceEuclideanCoord), end(DistanceEuclideanCoord), 0.0);
    double m =  sum / DistanceEuclideanCoord.size();
    
    double accum = 0.0;
    for_each (begin(DistanceEuclideanCoord), std::end(DistanceEuclideanCoord),[&](const double d){accum += (d - m) * (d - m);});
    double stdev = sqrt(accum / (DistanceEuclideanCoord.size()-1));

    double min_acceptable_value = m - stdev;
    double max_acceptable_value = m + stdev;

    // removing outliers and pushing the acceptable values to bounding boxes of the current frame.
    for(auto it2 = kptMatches.begin(); it2 != kptMatches.end(); it2++)
    {
        if(boundingBox.roi.contains(kptsCurr.at(it2->trainIdx).pt)) // checking if the keypoint is inside of the bounding box of the current frame.
        { 
            // calculating euclidean Distance with opencv function for each matched points
            double points_distance = cv::norm(kptsCurr.at(it2->trainIdx).pt - kptsPrev.at(it2->queryIdx).pt);

            if (( points_distance > min_acceptable_value ) && ( points_distance < max_acceptable_value))
            {
                boundingBox.kptMatches.push_back(*it2);
                boundingBox.keypoints.push_back(kptsCurr.at(it2->trainIdx));
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector <double> filtered_dist_ratio;
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // calculating mean and standar deviation.
    double sum = accumulate(begin(distRatios), end(distRatios), 0.0);
    double m =  sum / distRatios.size();
    
    double accum = 0.0;
    for_each (begin(distRatios), std::end(distRatios),[&](const double d){accum += (d - m) * (d - m);});
    double stdev = sqrt(accum / (distRatios.size()-1));

    double min_acceptable_value = m - stdev;
    double max_acceptable_value = m + stdev;

    
    for (int it3 = 0 ; it3 < distRatios.size(); ++it3)
    {
        if ((distRatios[it3]>min_acceptable_value) && ( distRatios[it3] < max_acceptable_value))
        {
            filtered_dist_ratio.push_back(distRatios[it3]);
        }
    }
    if (filtered_dist_ratio.size() == 0)
    {
        TTC = NAN;
        return;
    }  	
    std::sort(filtered_dist_ratio.begin(), filtered_dist_ratio.end());
    long medIndex = floor(filtered_dist_ratio.size() / 2.0);
    // compute median dist. ratio to remove outlier influence over the Filtered distance ratio vector.
    double medDistRatio = filtered_dist_ratio.size() % 2 == 0 ? (filtered_dist_ratio[medIndex - 1] + filtered_dist_ratio[medIndex]) / 2.0 : filtered_dist_ratio[medIndex];
    // compute median dist. ratio to remove outlier influence
    

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

double computeAverageLidarX(std::vector<LidarPoint> &lidarPoints){
    double average_x = 0;

    for(auto point : lidarPoints){
        average_x += point.x;
    }

    average_x /= lidarPoints.size();

    return average_x;
}

double computeMinLidarX(std::vector<LidarPoint> &lidarPoints){
    double min_x = 1e12;

    for(auto point : lidarPoints){
        if(point.x > 0.0 && point.x < min_x){
            min_x = point.x;
        }
    }

    return min_x;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    const double dt = 1/frameRate;

    double average_x_prev = computeAverageLidarX(lidarPointsPrev);
    double average_x_curr = computeAverageLidarX(lidarPointsCurr);

    double min_x_curr = computeMinLidarX(lidarPointsCurr);

    TTC = (min_x_curr * dt) / (average_x_prev - average_x_curr);
}

bool isInROI(cv::DMatch match, DataFrame dataframe, BoundingBox boundingBox, bool isPrevFrame){
    // check x position is in ROI

    double frame_x;
    double frame_y;

    if (isPrevFrame){
        frame_x = dataframe.keypoints.at(match.queryIdx).pt.x ;
        frame_y = dataframe.keypoints.at(match.queryIdx).pt.y ;
    }
    else{
        frame_x = dataframe.keypoints.at(match.trainIdx).pt.x ;
        frame_y = dataframe.keypoints.at(match.trainIdx).pt.y ;
    }

    if ( frame_x > (boundingBox).roi.x && frame_x < ((boundingBox).roi.x + (boundingBox).roi.width))
    {
        // check y position is in ROI
        if ( (frame_y >(boundingBox).roi.y) && (frame_y < ((boundingBox).roi.y + (boundingBox).roi.height)))
        {          
            return true;
        }
    }
    
    return false;
}

int findVectorOccurrences(vector<int> value)
{ 
    int index = 0;
    int highest = 0;
    for (unsigned int a = 0; a < value.size(); a++)
    {
        int count = 1;
        int Position = value.at(a);
        for (unsigned int b = a + 1; b < value.size(); b++)
        {
            if (value.at(b) == Position)
            {
                count++;
            }
        }
        if (count >= index)
        {
            index = count;
            highest = Position;
        }
    }
    return highest;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //initialize final result vector
    vector<vector<int>>final_result(currFrame.boundingBoxes.size());

    int bbox_curr_id = 0;
    int bbox_prev_id = 0;

    // iterate over matching points in order to create a vector with all distances
    for(auto match = matches.begin(); match != matches.end(); match++)
    {   
        // iterate over the prev frame bounding boxes
        for(auto prev_frame_bb = prevFrame.boundingBoxes.begin(); prev_frame_bb != prevFrame.boundingBoxes.end(); prev_frame_bb++)
        {        
            if(isInROI(*match, prevFrame, *prev_frame_bb, true)){
                bbox_prev_id = prev_frame_bb->boxID;
            }
        }
        
        for(auto curr_frame_bb = currFrame.boundingBoxes.begin(); curr_frame_bb != currFrame.boundingBoxes.end(); curr_frame_bb++)
        {         
            if(isInROI(*match, currFrame, *curr_frame_bb, false)){
                bbox_curr_id = curr_frame_bb->boxID;
            }
        }

        final_result[bbox_curr_id].push_back(bbox_prev_id);
      
    }
    
    // storing in bbBestMatches the pairs of the matching bounding boxes < previous , current >

    //cout << "the size of final vector of bbmatches is : " << final_result.size()<<endl;
    int bbox_current = 0;
    for(int it4 = 0 ; it4< final_result.size() ; it4++ )
    {
        int bBox_matching_prev = findVectorOccurrences(final_result[it4]);
        bbBestMatches.insert(pair<int, int>(bBox_matching_prev, bbox_current));
        //cout << bbox_current << ","<< bBox_matching_prev << endl;
        bbox_current ++;
    }
}
