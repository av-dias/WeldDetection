#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <list>
#include <fstream>
#include <numeric>
#include <filesystem>
#include <string>

using namespace std;
using namespace cv;
#define M_PI 3.14159265358979323846

//FUNCTION HEADERS
Mat noise_reduction(cv::Mat img, int degree);
Mat convert_negative(Mat img);
Mat convert_bw(Mat img);
Mat Dilation(Mat src);
Mat Erosion(Mat src);
Mat on_trackbar(Mat img);
int object_detection(Mat object_image);
void printVector(vector<int> v);
int check_intensity(Mat src);

vector<int> orderVector(vector<int> v, int n);
void MergeSortedIntervals(vector<int>& v, int s, int m, int e);
void MergeSort(vector<int>& v, int s, int e);

//ADJUSTABLE PARAMETERS
int degree_blur = 10; // DEFAULT: 10
int degree_bw = 205; // DEFAULT: 205
int size_threshold_min = 10000;
int size_threshold_max = 100000;
int elipse_tolerance = 80;
int rect_tolerance = 80;
int contrast = 1;
int brightness = 1;

//PARAMETERS
int dilation_elem = 2;
int dilation_size = 10;
int erosion_elem = 2;
int erosion_size = 3;

int threshval = 100;
string path_final = "C:/Users/av.dias/Desktop/ZDMP/zSteelTubes/store";
string path_test = "C:/Users/av.dias/Desktop/solda/";
string path_csv = "C:/Users/av.dias/Desktop/";

Mat dilation_dst;
Mat erosion_dst;

int DEBUG = 0; // 1-ON ; 0-OFF

namespace fs = std::filesystem;

void main()
{
    std::ofstream myfile;
    int filecount = 1; //increment this for naming purposes
    myfile.open(path_csv + "example_test.csv");
    vector<string> names;
    for (fs::recursive_directory_iterator i(path_final), end; i != end; ++i)
    {
        int nrNonBlackPixels=-1;
        if (!is_directory(i->path()) && i->path().extension()!=".html" && i->path().extension() != ".png") {
            Mat img;
            {
                stringstream f_negative, f_cropped, f_grey, f_bw_n, f_e, f_d, f_n_corrected;

                //DEFINE OUTPUT NAMES
                f_grey << path_test << "b" << i->path().filename().replace_extension("").string() << ".png";
                f_cropped << path_test << "c" << i->path().filename().replace_extension("").string() << ".png";
                f_negative << path_test << "d" << i->path().filename().replace_extension("").string() << ".png";
                f_n_corrected << path_test << "e" << i->path().filename().replace_extension("").string() << ".png";
                f_bw_n << path_test << "f" << i->path().filename().replace_extension("").string() << ".png";
                f_e << path_test << "g" << i->path().filename().replace_extension("").string() << ".png";
                f_d << path_test << "h" << i->path().filename().replace_extension("").string() << ".png";
                
                //READ IMAGE FROM LIST
                img = imread(i->path().string(), cv::IMREAD_COLOR);

                //cout << i->path().string() << endl;

                //NOISE REDUCTION
                img = noise_reduction(img, degree_blur);

                //CONVERT TO GREYSCALE
                cv::Mat grey_img;
                cv::cvtColor(img, grey_img, cv::COLOR_BGR2GRAY);
                if (DEBUG == 1)imwrite(f_grey.str().c_str(), grey_img);

                //CROP AND SAVE
                Mat cropped_image = grey_img(Range(0, 540), Range(335, 1015));
                if (DEBUG == 1)imwrite(f_cropped.str().c_str(), cropped_image);
                int image_size = cropped_image.rows * cropped_image.cols;
                int intensity = check_intensity(cropped_image);
                //if (DEBUG == 1)cout << "Intensity: " << intensity << " ";

                //CONVERT TO NEGATIVE AND SAVE
                Mat img_negative;
                //cropped_image.convertTo(img_negative, -1, contrast, brightness); //increase the brightness by 20
                img_negative = convert_negative(cropped_image);
                if (DEBUG == 1)imwrite(f_negative.str().c_str(), img_negative);

                //CONVERT TO BLACK AND WHITE FROM NEGATIVE
                Mat bw_image_n = convert_bw(img_negative);
                if (DEBUG == 1)imwrite(f_bw_n.str().c_str(), bw_image_n);
                
                nrNonBlackPixels = image_size - countNonZero(bw_image_n == 0);
                //CHECK IF IMAGE IS ALL BLACK, IF SO GO TO NEXT IMAGE
                if (nrNonBlackPixels == 0)    continue;
                //if (DEBUG == 1)cout << "Nr of black pixels: " << nrNonBlackPixels << " ";

                //EROSION
                Mat erosion_img = Erosion(bw_image_n);
                if (DEBUG == 1)imwrite(f_e.str().c_str(), erosion_img);

                filecount++;

                //Linked Components
                Mat object_image = on_trackbar(erosion_img);

                cv::cvtColor(object_image, object_image, cv::COLOR_BGR2GRAY);

                nrNonBlackPixels = image_size - countNonZero(object_image == 0);
                //CHECK IF IMAGE IS ALL BLACK, IF SO GO TO NEXT IMAGE
                if (nrNonBlackPixels == 0)    continue;

                //SHAPE DETECTION
                int isWeld = object_detection(object_image);
                if (isWeld >= elipse_tolerance)
                {
                    cout << "Object with " << isWeld << " ratio. Check " << i->path().string() << endl <<endl;
                    names.push_back(i->path().parent_path().string());
                    //if filename was not already inserted, insert
                    if(find(names.begin(), names.end(), i->path().parent_path().string()) == names.end())
                        myfile << i->path().parent_path().string() << endl;
                }
            }
        }
    }
    myfile.close();

    return;
}

int check_intensity(Mat src) {
    int intensity = 0;
    for (int i = 0; i < src.rows; i++) {
        for(int j=0; j<src.cols; j++)
        {
            intensity += (int)src.at<uchar>(i, j);
            //cout << intensity << ",";
        }
    }
    //cout << endl;
    //return intensity / (src.rows*src.cols);
    return intensity;
}

int object_detection(Mat object_image) {
    if (DEBUG == 1)imshow("Image with Object.", object_image);

    uint8_t* pixelPtr = (uint8_t*)object_image.data;
    int cn = object_image.channels();
    Scalar_<uint8_t> bgrPixel;

    int size_length = object_image.rows;       // declare the size of the vector
    vector<int> pixel_count_length(size_length, 0);   // create a vector to hold "size" int's
                                        // all initialized to zero

    int size_height = object_image.cols;       // declare the size of the vector
    vector<int> pixel_count_height(size_height, 0);   // create a vector to hold "size" int's
                                        // all initialized to zero

    //Calculate Image Lenghts
    for (int i = 0; i < object_image.rows; i++)
    {
        for (int j = 0; j < object_image.cols; j++)
        {
            bgrPixel.val[0] = pixelPtr[i * object_image.cols * cn + j * cn + 0]; // B
            bgrPixel.val[1] = pixelPtr[i * object_image.cols * cn + j * cn + 1]; // G
            bgrPixel.val[2] = pixelPtr[i * object_image.cols * cn + j * cn + 2]; // R

            if (bgrPixel.val[0] != 0 && bgrPixel.val[1] != 0 && bgrPixel.val[2] != 0)
            {
                pixel_count_length[i]++;
            }
        }

    }
    
    //Calculate Image Heigth
    for (int j = 0; j < object_image.cols ; j++)
    {
        for (int i = 0; i < object_image.rows; i++)
        {
            bgrPixel.val[0] = pixelPtr[i * object_image.cols * cn + j * cn + 0]; // B
            bgrPixel.val[1] = pixelPtr[i * object_image.cols * cn + j * cn + 1]; // G
            bgrPixel.val[2] = pixelPtr[i * object_image.cols * cn + j * cn + 2]; // R

            if (bgrPixel.val[0] != 0 && bgrPixel.val[1] != 0 && bgrPixel.val[2] != 0)
            {
                pixel_count_height[j]++;
            }
        }
    }

    //DELETE FROM VECTORS ALL ZEROs
    pixel_count_height.erase(std::remove(pixel_count_height.begin(), pixel_count_height.end(), 0), pixel_count_height.end());
    pixel_count_length.erase(std::remove(pixel_count_length.begin(), pixel_count_length.end(), 0), pixel_count_length.end());

    //REDECLARE VECTORS SIZEs
    size_length = pixel_count_length.size();
    size_height = pixel_count_height.size();

    //ORDER VECTORS
    pixel_count_height = orderVector(pixel_count_height, size_height);
    pixel_count_length = orderVector(pixel_count_length, size_length);

    //printVector(pixel_count_height);
    //printVector(pixel_count_length);

    // Assuming the axes of the ellipse are vertical/perpendicular.
    // A= Pi * raio_horizontal * r_vertical
    int A = pixel_count_height[size_height-1] / 2;
    int B = pixel_count_length[size_length - 1] / 2;

    double area_object = std::accumulate(pixel_count_length.begin(), pixel_count_length.end(), 0);
    double area_elipse = M_PI * A * B;
    double elipse_ratio = (area_object*100)/area_elipse;

    if (DEBUG == 1)cout << "Elipse Ratio: " << elipse_ratio << endl;

    if (elipse_ratio > elipse_tolerance)
    {
        int rect_ratio = (area_object*100) / ((2*A * 2*B));
        if (DEBUG == 1)cout << "Rect Ratio: " << rect_ratio << endl;
        if (rect_ratio > rect_tolerance)
        {
            elipse_ratio = 0;
        }
    }

    if (DEBUG == 1)waitKey();

    return elipse_ratio;
}

Mat on_trackbar(Mat img)
{
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for (int label = 1; label < nLabels; ++label) {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }
    Mat dst(img.size(), CV_8UC3);

    int size = nLabels;                 // declare the size of the vector
    vector<int> pixel_count(size, 0);   // create a vector to hold "size" int's
                                        // all initialized to zero

    // Write to the file
    //MyFile << "Files can be tricky; but it is fun enough!" << endl;
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
            pixel_count[label]++;
        }
    }

    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            
            // APAGAR AGORA OS COMPONENTES PEQUENOS size_threshold_min > x > size_threshold_max
            if  ((pixel_count[label] < size_threshold_min || pixel_count[label] > size_threshold_max))
            {
                pixel = Vec3b((0), (0), (0));
            }
        }
    }

    return dst;
}

Mat noise_reduction(cv::Mat img, int degree) {
    Mat result;
    blur(img, result, Size(degree, degree));

    return result;
}

Mat convert_negative(Mat img)
{
    return 255 - img;
}

Mat convert_bw(Mat img)
{
    return img > degree_bw;
}


Mat Erosion(Mat src)
{
    int erosion_type = 0;
    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(erosion_type,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size));
    erode(src, erosion_dst, element);
    //imshow("Erosion Demo", erosion_dst);
    return erosion_dst;
}


Mat Dilation(Mat src)
{
    int dilation_type = 0;
    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    dilate(src, dilation_dst, element);

    return dilation_dst;
}

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
void merge(int array[], int const left, int const mid, int const right)
{
    auto const subArrayOne = mid - left + 1;
    auto const subArrayTwo = right - mid;

    // Create temp arrays
    auto* leftArray = new int[subArrayOne],
        * rightArray = new int[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    auto indexOfSubArrayOne = 0, // Initial index of first sub-array
        indexOfSubArrayTwo = 0; // Initial index of second sub-array
    int indexOfMergedArray = left; // Initial index of merged array

    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo) {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]) {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne) {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo) {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
}

vector<int> orderVector(vector<int> v, int n)
{
    MergeSort(v, 0, n - 1);

    return v;
}

void MergeSortedIntervals(vector<int>& v, int s, int m, int e) {

    // temp is used to temporary store the vector obtained by merging
    // elements from [s to m] and [m+1 to e] in v
    vector<int> temp;

    int i, j;
    i = s;
    j = m + 1;

    while (i <= m && j <= e) {

        if (v[i] <= v[j]) {
            temp.push_back(v[i]);
            ++i;
        }
        else {
            temp.push_back(v[j]);
            ++j;
        }

    }

    while (i <= m) {
        temp.push_back(v[i]);
        ++i;
    }

    while (j <= e) {
        temp.push_back(v[j]);
        ++j;
    }

    for (int i = s; i <= e; ++i)
        v[i] = temp[i - s];

}

// the MergeSort function
// Sorts the array in the range [s to e] in v using
// merge sort algorithm
void MergeSort(vector<int>& v, int s, int e) {
    if (s < e) {
        int m = (s + e) / 2;
        MergeSort(v, s, m);
        MergeSort(v, m + 1, e);
        MergeSortedIntervals(v, s, m, e);
    }
}

void printVector(vector<int> v)
{
    for (int i = 0; i < v.size(); i++)
        cout << v[i] << " ";
    cout << endl << endl;
}