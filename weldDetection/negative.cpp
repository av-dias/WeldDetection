#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

void negative(Mat img)
{
    Mat img_negative = 255 - img;

    imshow("negative image", img_negative);
    //imwrite("C:/Users/av.dias/Desktop/Img/ns1.jpg", img_negative);

    return;
}