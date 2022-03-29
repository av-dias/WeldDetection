#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

const string path_img = "C:/Users/av.dias/Desktop/Img/b5.png";

int histogram(int argc, char** argv)
{
    Mat src = imread(path_img, IMREAD_COLOR);
    if (src.empty())
    {
        return EXIT_FAILURE;
    }
    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    int total = 0;
    for (int i = 1; i < histSize; i++)
    {
        //line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            //Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            //Scalar(255, 0, 0), 2, 8, 0);
        //line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            //Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            //Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    cout << cv::sum(g_hist)[0];
    cout << cv::sum(g_hist)[1];
    cout << cv::sum(g_hist)[2];
    imshow("Source image", src);
    imshow("calcHist Demo", histImage);
    waitKey();
    return EXIT_SUCCESS;
}