#pragma once
#include <opencv.hpp>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
typedef struct outInfo
{
    double angle;
    double centerX;
    double centerY;
    double longEdge;
    double shortEdge;

} outInfo;

typedef struct pointf
{
    float x, y;
    pointf(){ x = -1; y = -1; }
    pointf(float _x, float _y){ x = _x; y = _y; }
}pointf;

typedef struct rbox
{
    float x, y, w, h, a;
    rbox(){ x = 0; y = 0; w = 0; h = 0; a = 0; }
    rbox(float _x,float _y,float _w,float _h,float _a){ x = _x; y = _y; w = _w; h = _h; a = _a; }
}rbox;

typedef struct imgRbox
{
    rbox rb;
    int img_w, img_h;
    imgRbox(){ rb = rbox(); img_w = 0; img_h = 0; };
    imgRbox(rbox ra, int _w, int _h){ rb = ra; img_w = _w; img_h = _h; };
}imgRbox;

//////////////////////////////////////////////////////////////////////////
void img_rotate(Mat & imgIn, Mat &imgOut, Point cent, double dAngle, int fillVal);
void img_tans(Mat &imgIn, Mat &imgOut, int x, int y, int fillMode);
void rbox_rotate(rbox rboxIn, rbox rboxOut, Point center, float angle);
void draw_line2(Mat& img, pointf p0, pointf p1);

/////////////////////////////////////////////////////////////////////////
vector<string> splitString(string str,string splitChar);
vector<Point> drawRotRect1(Mat & img, Point p1, Point p2, Point p3);
RotatedRect  annotationString2RotRect(string str, Mat &img);
outInfo  drawCVRotRect(Mat & img, Point p1, Point p2, Point p3);  
Point2f onLineVerticalofP1P2(Point p1, Point p2, Point p3);

double getLongEdgeAngle(Point p1, Point p2, Point p3);
double Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[]);

int loadAnnotationAndShow(string annotationPath,Mat &img);
int loadAnnotation_getVerShow(string annotationPath, Mat &img);
void getRboxVertices(rbox ra, pointf * vertices);
void showRbox(Mat & img, rbox ra);
void showIntersection(string annotation ,Mat & img);

