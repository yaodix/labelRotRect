
#include <string>
#include <iostream>
#include <fstream>
#include "labelRotRect.h"
using namespace cv;
using namespace std;


Mat  srcImg, maskImg, srcPatch, maskPatch;
int halfSize = 128;
static int numCnt = 0;
string imgToTxtPath;
int imgCnt = 0;
static int draw_symbol = 0;

static void onMouseSrcImg(int event, int x, int y, int, void* val)
{
    static Point longEdgePt1;
    static Point longEdgePt2;
    static  Point thirdPt;
    int longEdge = 0, shortEdge = 0;
    int num = *(int*)val;
    string imgCntInfo = format("its  %d  pic , all %d", num+1, imgCnt);

     if (event == cv::EVENT_LBUTTONUP)
    {
        switch (draw_symbol)
        {
        case  0:
            draw_symbol = 1;
            longEdgePt1 = Point(x, y);
            break;
        case 1:
            draw_symbol = 2;
            longEdgePt2 = Point(x, y);
            break;
        case 2:
            draw_symbol = 3;
            thirdPt = Point(x, y);
            break;

        default:
            break;
        }

        char srcSavePath[100];
        char maskSavePath[100];
        //sprintf_s(srcSavePath, "F:\\MyData\\CrackForest-dataset-master\\myCrack\\image\\%3d.jpg", numCnt);
        //sprintf_s(maskSavePath, "F:\\MyData\\CrackForest-dataset-master\\myCrack\\label\\%3d.jpg", numCnt);
        numCnt++;
        //imwrite(srcSavePath, dst);
        //imwrite(maskSavePath, dstMask);

    }
    if (event == cv::EVENT_MOUSEMOVE)
    {
        Mat showRectImg;
        showRectImg = srcImg.clone();
        if (draw_symbol == 3)
        {
            draw_symbol = 0;
           vector<Point>  vecpts = drawRotRect1(showRectImg, longEdgePt1, longEdgePt2, thirdPt);
           outInfo real = drawCVRotRect(showRectImg, vecpts[0],vecpts[1],vecpts[2]);
           cout << "Angle = " << real.angle<< "( " << real.centerX <<","<< real.centerY <<")"
               << real.longEdge << "x" << real.shortEdge << endl;
           circle(showRectImg, Point(real.centerX, real.centerY), 2, Scalar(0, 0, 254), 3);

           string str = std::to_string(real.angle);
           str.resize(5);
            cv:String strAngle = cv::String(str);
           putText(showRectImg, strAngle, Point(real.centerX, real.centerY), 3, 1.5, Scalar(0, 0,250));

           putText(srcImg, imgCntInfo, Point(20, 40), 2, 1.0, Scalar(0, 0, 250));

           //输出中心点，长短边，角度
          
           fstream output(imgToTxtPath,std::ios::app);
           string outCenterX = to_string(double(real.centerX)/srcImg.cols);
           outCenterX.resize(8);
           string outCenterY = to_string(double(real.centerY)/srcImg.rows);
           outCenterY.resize(8);
           string outLongEdge = to_string(double(real.longEdge)/srcImg.cols);  // longEdge ---> widht
           outLongEdge.resize(8);
           string outShortEdge = to_string(double(real.shortEdge) / srcImg.rows);   //
           outShortEdge.resize(8);
           string outAngle = to_string(real.angle/180.*CV_PI);
           outAngle.resize(8);

           output << "0 " << outCenterX << " " << outCenterY << " " << outLongEdge << " " 
                                    << outShortEdge << " " << outAngle << endl;

            srcImg = showRectImg.clone();
        }
        if (draw_symbol == 1)
        {
            line(showRectImg, longEdgePt1, Point(x, y), Scalar(0, 255, 0), 3);
        }
        if (draw_symbol == 2)
        {
            drawRotRect1(showRectImg, longEdgePt1, longEdgePt2, Point(x, y));
        }


        imshow("srcImgWin", showRectImg);
    }
}



int main()
{
    namedWindow("srcImgWin", WINDOW_NORMAL);

    //添加标注路径
    cv::String srcPathes = R"(C:\MyData\pen\*.png)";
    Mat toSave;
    vector<cv::String> vecSrcPathes;
    glob(srcPathes, vecSrcPathes);
    imgCnt = vecSrcPathes.size();
    for (int i =0; i < vecSrcPathes.size(); i++)
    {

        srcImg = imread(vecSrcPathes[i], cv::IMREAD_COLOR);
        toSave = srcImg;
        if (srcImg.empty())
        {
            cout << "image doesnt exist!!" << endl;
            return 0;
        }
        int dotPos = vecSrcPathes[i].rfind('.');
        string strTemp = vecSrcPathes[i].substr(0, dotPos);
        imgToTxtPath = strTemp + ".txt";
        string imgCntInfo = format(" %d  pic , all %d", i+1, imgCnt);
        putText(srcImg, imgCntInfo, Point(10, 10), 2, 0.5, Scalar(0, 0, 250));
        cout << vecSrcPathes[i] << endl;
        string txtFile = vecSrcPathes[i];
        txtFile.resize(txtFile.size() - 3);
        txtFile += "txt";
        //loadAnnotationAndShow(txtFile, srcImg);
        loadAnnotation_getVerShow(txtFile, srcImg);
        Mat bitIOUImg;
        //showIntersection(txtFile, srcImg);
        //showIOU_bitwise(txtFile, bitIOUImg);

        imshow("srcImgWin", srcImg);

        cv::setMouseCallback("srcImgWin", onMouseSrcImg, &i);

        int key;
        while (true)
        {        
            key = waitKey(0);
            switch (key)
            {
                case 32:
                case 'd':
                    if (i == vecSrcPathes.size() - 1)  //最后一张
                    {
                        i = -1;
                    }
                    break;
                case 'a':                
                        i = i - 2;
                        if (i < -1)
                        {
                            i = vecSrcPathes.size() - 2;
                        }
                    break;

                case 27: //Esc
                    draw_symbol =0;
                    i = i - 1;    //当前图片不变
    
                    break;
            }
            break;
        }
    }
    
    return 0;
}