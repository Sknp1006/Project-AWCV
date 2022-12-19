#include "CoreTest.hpp"
#include <opencv2/highgui.hpp>



int main(int argc, char *argv[])
{
    // std::cout << "欢迎使用算法测试工具，请开始你的表演！" << std::endl;
    fprintf(stdout, "欢迎使用算法测试工具，请开始你的表演！\n");
    // cv::Mat img = cv::imread(R"(..\data\images\025.jpg)");
    // // cv::imshow("img", img);
    // cv::Mat temp;
    // initGrayMappingTrackbar();
    // while (cv::waitKey(1)!=27)
    // {
    //     //t_gammaImage(img, temp);
    //     //t_autoGammaImage(img, temp);
    //     t_zoomGray(img, temp);
    // }

    testFaceDetectorDNN();//人脸检测（OK）
}