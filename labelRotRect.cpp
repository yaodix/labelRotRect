
#include "labelRotRect.h"

//计算IOU
//计算IOU
void getRboxVertices(rbox ra, pointf * vertices)
{
    //矩形中心为原点的正矩形顶点坐标
    pointf pts[4];    //左上，右上，右下，左下顺序
    pts[0].x = -ra.w / 2;
    pts[0].y = ra.h / 2;
    pts[1].x = ra.w / 2;
    pts[1].y = ra.h / 2;

    pts[2].x = ra.w / 2;
    pts[2].y = -ra.h / 2;
    pts[3].x = -ra.w / 2;
    pts[3].y = -ra.h / 2;

    float thelta = ra.a;  //换算后的角度
    float x2 = ra.x;
    float y2 = ra.y;
    for (int i = 0; i < 4; i++)
    {
        float x1 = pts[i].x;
        float y1 = pts[i].y;
        float x = (x1)*cos( thelta) - (y1)*sin( thelta);
        float y = (x1)*sin(thelta) + (y1)*cos(thelta);

        vertices[i].x = x2 + x;
        vertices[i].y = y2 - y;
    }
}

// 计算交点，(-1,-1)没有交点 
pointf intersectionOf2SegLine(pointf p1, pointf p2, pointf p3, pointf p4)
{
    pointf ptNull, ptRes;
    ptNull.x = -1;
    ptNull.y = -1;

    float x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
    float y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;
    double denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if (denom == 0.0) { // Lines are parallel.
        return ptNull;
    }
    double ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    double ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
    if (ua >= 0.0f && ua <= 1.0f && ub >= 0.0f && ub <= 1.0f) {
        // Get the intersection point.
        ptRes.x = (x1 + ua*(x2 - x1));
        ptRes.y = (y1 + ua*(y2 - y1));
        return  ptRes;
    }
    return ptNull;
}

int  isInsideRbox(pointf pt, rbox ra)
{
    pointf ver[4];
    pointf center;
    center.x = ra.x;
    center.y = ra.y;
    getRboxVertices(ra, ver);
    for (int i = 0; i < 4; i++)
    {
        pointf sec = intersectionOf2SegLine(center, pt, ver[i], ver[(i + 1) % 4]);
        if (sec.x >0)
        {
            return 0;  // 有交点不在矩形内
        }
    }
    return 1;
}

float triangleArea(pointf p1, pointf p2, pointf p3)
{
    float s = ((p3.x - p1.x)*(p2.y - p1.y) - (p2.x - p1.x)*(p3.y - p1.y)) / 2.0;
    return s > 0 ? s : -s;
}

vector<string> splitString(string str, string splitChar)
{
    string::iterator strIte = str.begin();
    string::iterator firstPos;
    vector<string> vecRes;
    while (true)
    {
        firstPos = find_first_of(strIte, str.end(), splitChar.begin(), splitChar.end());
        vecRes.push_back(str.substr(distance(str.begin(), strIte), std::distance(strIte, firstPos)));
        if (firstPos == str.end())
        {
            break;
        }
        else
        {
            strIte = firstPos + splitChar.size();

        }
    }

    return vecRes;
}

void img_rotate(Mat & imgIn, Mat &imgOut, Point cent, double dAngle, int fillVal)
{

    //计算二维旋转的仿射变换矩阵  
    cv::Mat M = cv::getRotationMatrix2D(cent, dAngle, 1);

    //变换图像，并用黑色填充其余值
    //cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 
    cv::warpAffine(imgIn, imgOut, M, imgIn.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS, 0, cv::Scalar(fillVal));
}

void img_tans(Mat &imgIn, Mat &imgOut, int x, int y, int fillMode)
{
    cv::Size dst_sz = imgIn.size();

    //定义平移矩阵
    cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);

    t_mat.at<float>(0, 0) = 1;
    t_mat.at<float>(0, 2) = x; //水平平移量
    t_mat.at<float>(1, 1) = 1;
    t_mat.at<float>(1, 2) = y; //竖直平移量

    //根据平移矩阵进行仿射变换
    cv::warpAffine(imgIn, imgOut, t_mat, dst_sz);

}

//返回角度
double get2VecAngle(pointf pOrigin, pointf p1, pointf p2)  // 
{
    double dLineDirVec1[3];
    double dLineDirVec2[3];

    dLineDirVec1[0] = p1.x - pOrigin.x;
    dLineDirVec1[1] = p1.y - pOrigin.y;
    dLineDirVec1[2] = 0;

    dLineDirVec2[0] = p2.x - pOrigin.x;
    dLineDirVec2[1] = p2.y - pOrigin.y;
    dLineDirVec2[2] = 0;

    return (acos((dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
        / (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
        *sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2])))) / CV_PI * 180.0;
}
void rbox_rotate(imgRbox rboxIn, imgRbox &rboxOut, Point center, float angle)
{
    rbox ra = rboxIn.rb;
    pointf verIn[4];
    pointf verOut[4];
    int row = rboxIn.img_h;
    int col = rboxIn.img_w;
    getRboxVertices(ra, verIn);
    for (int i = 0; i < 4;i++)
    {
        pointf pt = verIn[i];
        float x1 = pt.x;
        float y1 = row - pt.y;
        float x2 = center.x;
        float y2 = row - center.y;
        float x = (x1 - x2)*cos(angle) - (y1 - y2)*sin(angle) + x2;
        float y = (x1 - x2)*sin( angle) + (y1 - y2)*cos (angle) + y2;
        x = x;
        y = row - y;
        verOut[i] = pointf(x, y);
    }

    float newx = (verOut[0].x + verOut[2].x) / 2.;
    float newy = (verOut[0].y + verOut[2].y) / 2.;
    if (newx < 0 || newx > col || newy < 0 || newy > row)
    {
        rboxOut =  imgRbox(rbox(), 0, 0);
        return;
    }
    float newa = get2VecAngle(verOut[1], verOut[0], pointf(verOut[1].x+4, verOut[1].y));
    rbox newrb(newx, newy, rboxIn.rb.w, rboxIn.rb.h, newa);
    rboxOut =imgRbox( newrb,rboxIn.img_w,rboxIn.img_h);

}

void draw_line2(Mat& img, pointf p0, pointf p1)
{
    int x0 = p0.x;
    int y0 = p0.y;
    int x1 = p1.x;
    int y1 = p1.y;
    int deltaX = 0;
    int deltaY = 0;

    int xTemp = 0;
    int yTemp = 0;
    deltaX = (x1 - x0);
    deltaY = (y1 - y0);
    int startx = 0;
    int starty = 0;
    int endx = 0;
    int endy = 0;
    if (abs(x0 - x1) > abs(y0 - y1))
    {
        x0 < x1 ? (startx = x0, endx = x1, starty = y0, endy = y1) : (startx = x1, endx = x0, starty = y1, endy = y0);
        deltaX = endx - startx;
        deltaY = endy - starty;
        if (startx <1)
            startx = 1;
        if (endx > img.cols - 2)
        {
            endx = img.cols - 2;
        }
        for (int i = startx; i <= endx; i++)
        {
            yTemp = double(i - startx) / deltaX*deltaY + starty;
            if (yTemp < 1)
                yTemp = 1;
            if (yTemp > img.rows - 2)
            {
                yTemp = img.rows - 2;
            }
            img.at<Vec3b>(yTemp, i) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp - 1, i - 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp - 1, i) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp - 1, i + 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp, i - 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp, i + 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp + 1, i + 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp + 1, i) = Vec3b(0, 0, 255);
            img.at<Vec3b>(yTemp + 1, i - 1) = Vec3b(0, 0, 255);

        }

    }
    else
    {
        y0 < y1 ? (startx = x0, endx = x1, starty = y0, endy = y1) : (startx = x1, endx = x0, starty = y1, endy = y0);
        deltaX = endx - startx;
        deltaY = endy - starty;
        if (starty <1)
            starty = 1;
        if (endy > img.rows - 2)
        {
            endy = img.rows - 2;
        }
        for (int j = starty; j <= endy; j++)
        {
            xTemp = abs(double(j - starty) / deltaY)*deltaX + startx;
            if (xTemp < 1)
                xTemp = 1;
            if (xTemp > img.cols - 2)
            {
                xTemp = img.cols - 2;
            }
            img.at<Vec3b>(j - 1, xTemp - 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j - 1, xTemp) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j - 1, xTemp + 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j, xTemp - 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j, xTemp) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j, xTemp + 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j + 1, xTemp - 1) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j + 1, xTemp) = Vec3b(0, 0, 255);
            img.at<Vec3b>(j + 1, xTemp + 1) = Vec3b(0, 0, 255);
        }
    }
}

// 求两空间向量的夹角，返回值单位：弧度
double Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[])
{
    return acos((dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
        / (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
        *sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2])));
}

float rbox_intersection(rbox ra, rbox rb,Mat &img)
{
    pointf joint;
    pointf joints[30];
    pointf vertices_ra[4];
    pointf vertices_rb[4];
    int cnt = 0;
    getRboxVertices(ra, vertices_ra);
    getRboxVertices(rb, vertices_rb);

    //获取边的交点
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            joint = intersectionOf2SegLine(vertices_ra[i], vertices_ra[(i + 1) % 4], vertices_rb[j], vertices_rb[(j + 1) % 4]);
            if (joint.x >0)
            {
                joints[cnt++] = joint;
            }
        }
    }
    //查看顶点是否在矩形内
    for (int i = 0; i < 4; i++)
    {
        if (isInsideRbox(vertices_ra[i], rb))
        {
            joints[cnt++] = vertices_ra[i];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        if (isInsideRbox(vertices_rb[i], ra))
        {
            joints[cnt++] = vertices_rb[i];
        }
    }

    //对点进行排序
    pointf joints_center, joint_center_Yminus;
    float angles[30];
    for (int i = 0; i < cnt; i++)
    {
        joints_center.x += joints[i].x;
        joints_center.y += joints[i].y;
    }
    joints_center.x = joints_center.x / cnt;
    joints_center.y = joints_center.y / cnt;
    joint_center_Yminus.x = joints_center.x;
    joint_center_Yminus.y = joints_center.y - 0.1;

    for (int i = 0; i < cnt; i++)
    {
        double angle_clock = get2VecAngle(joints_center, joint_center_Yminus, joints[i]);
        if (joints[i].x >= joints_center.x)
        {
            angles[i] = angle_clock;
        }
        else
        {
            angles[i] = 360 - angle_clock;
        }
    }

    //排序,从小到大 即顺时针
    for (int i = 0; i < cnt; i++)
    {
        for (int j = i + 1; j < cnt; j++)
        {
            if (angles[i] > angles[j])
            {
                double temp = angles[i];
                angles[i] = angles[j];
                angles[j] = temp;

                pointf tempPt;
                tempPt.x = joints[i].x;
                tempPt.y = joints[i].y;
                joints[i].x = joints[j].x;
                joints[i].y = joints[j].y;
                joints[j].x = tempPt.x;
                joints[j].y = tempPt.y;

            }
        }
    }
    ///*
    for (int i = 0; i < cnt; i++)
    {
    circle(img, Point(joints[i].x, joints[i].y), 3, Scalar(0, 250, 5), 5);
    putText(img, to_string(i), Point(joints[i].x, joints[i].y), 1.0, 3, Scalar(0, 250, 5), 3);
    }
    //*/

    //计算面积
    float areaSum = 0;
    for (int i = 1; i < cnt - 1; i++)
    {
        float area = triangleArea(joints[0], joints[i], joints[i + 1]);
         cv::line(img, Point(joints[0].x, joints[0].y), Point(joints[i + 1].x, joints[i + 1].y), Scalar(255, 0, 0), 3);
         cv::line(img, Point(joints[0].x, joints[0].y), Point(joints[i].x, joints[i].y), Scalar(255, 0, 0), 3);
           cv::line(img, Point(joints[i].x, joints[i].y), Point(joints[i + 1].x, joints[i + 1].y), Scalar(255, 0, 0), 3);

        areaSum += area;
    }
    return areaSum;
}


void showRbox(Mat & img, rbox ra)
{
    pointf ver[4];
    getRboxVertices(ra, ver);
    for (int i = 0; i < 4; i++)
    {
        //line(img, Point(ver[i].x, ver[(i) % 4].y),
          //  Point(ver[(i + 1) % 4].x, ver[(i + 1) % 4].y), Scalar(0, 0, 255), 3);
        draw_line2(img, pointf(ver[i].x, ver[(i) % 4].y),
            pointf(ver[(i + 1) % 4].x, ver[(i + 1) % 4].y));
    }
}

void fillRbox(Mat & img, rbox ra)
{
    pointf ver[4];
    getRboxVertices(ra, ver);
    for (int i = 0; i < 4; i++)
    {
        line(img, Point(ver[i].x, ver[(i) % 4].y),
            Point(ver[(i + 1) % 4].x, ver[(i + 1) % 4].y), Scalar(255, 255, 255), 1);
    }

    floodFill(img, Point(ra.x, ra.y), Scalar(255, 255, 255));
}


//***************************↑
vector<Point> drawRotRect1(Mat & img, Point p1, Point p2, Point p3)
{


        p3 = onLineVerticalofP1P2(p1, p2, p3);
        int xOffset = p3.x - p2.x;
        int yOffset = p3.y - p2.y;
        Point p4(p1.x + xOffset, p1.y + yOffset);

        line(img, p1, p2, Scalar(0, 255, 0),3);
        line(img, p3, p2, Scalar(0, 255, 0),3);
        line(img, p3, p4, Scalar(0, 255, 0),3);
        line(img, p1, p4, Scalar(0, 255, 0),3);

        return vector<Point>{p1, p2, p3, p4};

}


outInfo   drawCVRotRect(Mat & img, Point p1, Point p2, Point p3) //angle为0-180
{
    outInfo info;
        Point2f center = (p1 + p3) / 2;

        double angle = getLongEdgeAngle(p1, p2, p3);

        //转换为opencv表示
        RotatedRect rect;
        if (true)
        {

                if (0 <= angle && angle <= 90)
                {
                    double cvAngel = -(angle);
                    double distance = powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2);
                    distance = sqrtf(distance);
                    int width = distance;
                    int height = sqrtf(powf((p3.x - p2.x), 2) + powf((p3.y - p2.y), 2));

                    //长边为width
                    if (width > height)
                        rect = RotatedRect(center, Size2f(width, height), cvAngel);
                    else
                    {
                        rect = RotatedRect(center, Size2f(height, width), cvAngel);
                    }

                }
                else
                {
                    double cvAngel = -((angle - 90));
                    //p1p2默认为长边
                    double distance = powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2);
                    distance = sqrtf(distance);
                    int width = distance;
                    int height = sqrtf(powf((p3.x - p2.x), 2) + powf((p3.y - p2.y), 2));
                    //短边为width
                    if (width < height)
                        rect = RotatedRect(center, Size2f(width, height), cvAngel);
                    else
                    {
                        rect = RotatedRect(center, Size2f(height, width), cvAngel);
                    }
                }

                Point2f pts[4];
                rect.points(pts);

                for (int i = 0; i < 4; i++)
                {
                    line(img, pts[i], pts[(i + 1) % 4], Scalar(0, 0,250),3);
                }
        }
        info.angle = angle;
        info.centerX = center.x;
        info.centerY = center.y;

        int w = rect.size.width;
        int h = rect.size.height;
        if (h >w)
        {
            info.longEdge = h;
            info.shortEdge = w;
        } 
        else
        {
            info.longEdge = w;
            info.shortEdge = h;
        }
        
        return info;
    }

Point2f  onLineVerticalofP1P2(Point p1, Point p2, Point p3)
{

    if (p1.y == p2.y)
    {
        return Point2f(p2.x, p3.y);
    }
    double k = double(p2.y - p1.y) / ((p2.x - p1.x)+0.1);
    double vk = -1 / k;
    double b = double(p2.y) - p2.x*vk;
        
    if (vk > 4.5 || vk <-7 )
    {
        double ptx = (p3.y - b) / vk;
        return Point2f(ptx, p3.y);
    }
    else
    {
    double pty = vk*p3.x + b;
    return Point2f(p3.x, pty);

    }


}

double getLongEdgeAngle(Point p1, Point p2, Point p3)
{
    Point startPt, endPt;    //startPt 在图像下面
    if (abs(p2.y - p1.y) + abs(p2.x - p1.x) > abs(p2.y - p3.y) + abs(p2.x - p3.x)) //找出长边
    {
        if (p1.y > p2.y)
        {
            startPt = p1;
            endPt = p2;
        } 
        else
        {
            startPt = p2;
            endPt = p1;
        }

    } 
    else
    {
        if (p3.y > p2.y)
        {
            startPt = p3;
            endPt = p2;
        }
        else
        {
            startPt = p2;
            endPt = p3;
        }

    }

    // calc ange with horiz  line
    Point endHpt(startPt.x + 10, startPt.y);

    double dA[3], dB[3];
    dA[0] = startPt.x - endPt.x;
    dA[1] = startPt.y - endPt.y;
    dA[2] = 0;
    dB[0] = startPt.x - endHpt.x;
    dB[1] = startPt.y - endHpt.y;
    dB[2] = 0;

    return  Get2VecAngle(dA, dB) / CV_PI * 180; //输出到脚本以角度表示
}

//加载标记文件并显示
int loadAnnotationAndShow(string annotationPath,Mat &img)
{
    char strLine[100];
    vector<string> vecStr;
    vector<RotatedRect> vecRRect;
    ifstream fin(annotationPath);
    if (!fin) return-1;

    while (fin.getline(strLine, 100))
    {
        vecStr.push_back(string(strLine));
    }
    for (int i = 0; i < vecStr.size();i++)
    {
        vecRRect.push_back(annotationString2RotRect(vecStr[i],img));
    }

    fin.close();
    for (int i = 0; i < vecRRect.size();i++)
    {
        Point2f pts[4];
        vecRRect[i].points(pts);
        putText(img, to_string(i), (pts[0] + pts[2]) / 2, 1, 2.5, Scalar(255, 0, 255),4);
        for (int i = 0; i < 4; i++)
        {
            line(img, pts[i], pts[(i + 1) % 4], Scalar(0, 0, 250), 3);
        }
    }
}



//显示计算的三角区域
void showIntersection(string txtFile,Mat & img)
{

    Mat rb1img = img.clone(), rb2img = img.clone();
    rb1img.setTo(0);
    rb2img.setTo(0);
    char strLine[100];
    vector<string> vecStr;
    vector<RotatedRect> vecRRect;
    ifstream fin(txtFile);
    if (!fin) return ;
    rbox rb[10];

    while (fin.getline(strLine, 100))
    {
        vecStr.push_back(string(strLine));
    }
    if (vecStr.size() ==0)
    {
        return;
    }
    for (int i = 0; i < vecStr.size(); i++)
    {
            vector<string> vecSplit = splitString(vecStr[i], " ");
            rb[i].x = atof(vecSplit[1].c_str())*img.cols;
            rb[i].y = atof(vecSplit[2].c_str())*img.rows;
            rb[i].w = atof(vecSplit[3].c_str())*img.cols;
            rb[i].h = atof(vecSplit[4].c_str())*img.rows;
            rb[i].a = atof(vecSplit[5].c_str());
    }
    fin.close();

    fillRbox(rb1img, rb[0]);
    fillRbox(rb2img, rb[1]);
    Mat dst, binDst;
    bitwise_and(rb1img, rb2img, dst);
    cvtColor(dst, binDst, cv::COLOR_BGR2GRAY);
    int bitArea = countNonZero(binDst);

    float inter = rbox_intersection(rb[0], rb[1], img);

    cout << "triangle IOU       =  " << inter << endl;
    cout << "bitwise IOU        =  " << bitArea << endl;
    cout << " IOU difference  = " << abs(inter - bitArea) << endl;
}

int loadAnnotation_getVerShow(string annotationPath, Mat &img)
{
    char strLine[100];
    vector<string> vecStr;
    vector<RotatedRect> vecRRect;
    ifstream fin(annotationPath);
    if (!fin) return-1;
        rbox rb[10];

    while (fin.getline(strLine, 100))
    {
        vecStr.push_back(string(strLine));
    }

        for (int i = 0; i < vecStr.size(); i++)
        {
            vector<string> vecSplit = splitString(vecStr[i], " ");
            rb[i].x = atof(vecSplit[1].c_str())*img.cols;
            rb[i].y = atof(vecSplit[2].c_str())*img.rows;
            rb[i].w = atof(vecSplit[3].c_str())*img.cols;
            rb[i].h = atof(vecSplit[4].c_str())*img.rows;
            rb[i].a = atof(vecSplit[5].c_str());
        }
    fin.close();

    for (int i = 0; i < vecStr.size(); i++)
    {
        showRbox(img, rb[i]);
    }

}
cv::RotatedRect annotationString2RotRect(string str, Mat &img)
{
    vector<string> annoLine = splitString(str, " ");
    int  centerX = atof(annoLine[1].c_str()) *img.cols;
    int centerY = atof(annoLine[2].c_str())*img.rows;

    int longEdge = atof(annoLine[3].c_str())*img.cols;
    int shortEdge = atof(annoLine[4].c_str())*img.rows;

    double angle = atof(annoLine[5].c_str());
    angle = angle*180. / CV_PI;
    if (angle <= 90)
    {
        return RotatedRect(Point(centerX, centerY), Size(longEdge, shortEdge), -angle);
    } 
    else
    {
        return RotatedRect(Point(centerX, centerY), Size(shortEdge, longEdge), -angle+90);
    }

}
vector<imgRbox> readAnnotation(string annotationPath)
{
    char strLine[100];
    vector<string> vecStr;
    vector<imgRbox> vecImgRbox;
    imgRbox rb;
    Mat img;

    ifstream fin(annotationPath);
    if (!fin) return  vector<imgRbox>();

    annotationPath.resize(annotationPath.size() - 3);
    annotationPath += "bmp";
    img = imread(annotationPath);
    if (img.empty())
    {
        return vector<imgRbox>();
    }
    while (fin.getline(strLine, 100))
    {
        vecStr.push_back(string(strLine));
    }

    for (int i = 0; i < vecStr.size(); i++)
    {
        vector<string> vecSplit = splitString(vecStr[i], " ");
        rb.rb.x = atof(vecSplit[1].c_str())*img.cols;
        rb.rb.y = atof(vecSplit[2].c_str())*img.rows;
        rb.rb.w = atof(vecSplit[3].c_str())*img.cols;
        rb.rb.h = atof(vecSplit[4].c_str())*img.rows;
        rb.rb.a = atof(vecSplit[5].c_str());
        rb.img_w = img.cols;
        rb.img_h = img.rows;
        vecImgRbox.push_back(rb);
    }
    fin.close();

    return vecImgRbox;
}

vector<string> rot_enhance(string annotationPath, Mat &newImg,float angle)
{
    string newAnnotatioin;
    vector<imgRbox> vecImgRbox;
    vector<imgRbox> vecImgRboxOut;
    vecImgRbox = readAnnotation(annotationPath);
    newAnnotatioin = annotationPath;
    annotationPath.resize(annotationPath.size() - 3);
    annotationPath += "bmp";
    Mat img = imread(annotationPath);
    pointf rot_center(0,0);
    if (vecImgRbox.size() < 1)return vector<string>();
    for (auto box:vecImgRbox)
    {
        rot_center.x += box.rb.x;
        rot_center.y += box.rb.y;
    }
    rot_center.x = rot_center.x / vecImgRbox.size();
    rot_center.y = rot_center.y / vecImgRbox.size();
    Mat transImg;
    img_tans(img, transImg, (vecImgRbox[0].img_w/2 - rot_center.x), (vecImgRbox[0].img_h/2 - rot_center.y), 118);
    //移动标注框
    for (auto box:vecImgRbox)
    {
        box.rb.x = box.rb.x + (vecImgRbox[0].img_w / 2 - rot_center.x);
        box.rb.y = box.rb.y + (vecImgRbox[0].img_h / 2 - rot_center.y);

    }

    Mat rot_img;
    img_rotate(transImg, rot_img, Point(img.cols/2,img.rows/2),angle,118);
    //旋转标注并输出
    for (int i = 0; i < vecImgRbox.size(); i++)
    {
        imgRbox temp;
        rbox_rotate(vecImgRbox[i], temp, Point(vecImgRbox[0].img_w/2,vecImgRbox[0].img_h/2),angle);
        if (temp.img_w != 0)
        {
            vecImgRboxOut.push_back(temp);
        }
    }

    newAnnotatioin.resize(newAnnotatioin.size() - 4);
    newAnnotatioin += "_" + to_string(angle) + ".txt";
    ofstream fout(newAnnotatioin);
    vector<string> annoOut;
    for (int i = 0; i < vecImgRboxOut.size();i++)
    {
        string str = "0 " + to_string(vecImgRboxOut[i].rb.x/newImg.cols )+ " " + to_string(vecImgRboxOut[i].rb.y/newImg.rows) + " " +
            to_string(vecImgRboxOut[i].rb.w /newImg.cols)+ " " +to_string(vecImgRboxOut[i].rb.h /newImg.rows)+ " " + to_string(vecImgRboxOut[i].rb.a );
        annoOut.push_back(str);
    }
    fout.close();
    return annoOut;
}
