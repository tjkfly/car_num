#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define DEBUG
using namespace std;
using namespace cv;
using namespace cv::dnn;

string label_list[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
              "C", "´¨",
              "D", "E", "¶õ", "F",
              "G", "¸Ó", "¸Ê", "¹ó", "¹ð",
              "H", "ºÚ", "»¦",
              "J", "¼½", "½ò", "¾©", "¼ª",
              "K", "L", "ÁÉ", "Â³", "M", "ÃÉ", "Ãö",
              "N", "Äþ",
              "P", "Q", "Çà", "Çí",
              "R", "S", "ÉÂ", "ËÕ", "½ú",
              "T", "U", "V", "W", "Íî",
              "X", "Ïæ", "ÐÂ",
              "Y", "Ô¥", "Óå", "ÔÁ", "ÔÆ",
              "Z", "²Ø", "Õã"};
int main()
{
    string  model_path = "../infer/frozen_graph.pb";
    Net net = readNetFromTensorflow(model_path);
    Mat license_plate = imread("E:\\code\\tf\\proj\\car_num\\zf.png",1);
    int image_h = license_plate.rows;
//    cout<<image_h<<endl;
    Mat gray_plate;
    Mat binary_plate;
    cvtColor(license_plate,gray_plate,COLOR_BGR2GRAY);
    threshold(gray_plate,binary_plate,175,255,THRESH_OTSU);


    vector<uint> pix(binary_plate.cols,0);

    for(int i = 0; i < binary_plate.cols;i++)
    {
        pix.push_back(0);
        for(int j = 0;j<binary_plate.rows;j++)
        {
           pix[i] = pix[i] +  binary_plate.at<uchar>(j,i);
        }
//        printf("%d ",pix[i]);
//        cout<<endl;
    }

    uint i = 0;
    uint num = 0;
    vector<Point> index_range;
    while(i < pix.size())
    {
        if (pix[i] == 0)
        {
            i +=1 ;
        }
        else
        {
            uint index = i + 1;
            while(pix[index] != 0)
            {
                index += 1;
            }
            index_range.push_back(Point(i,index-1));
            num += 1;
            i = index;
        }
    }
//    cout<<index_range.size();
    vector<Mat> seg_img;
    for(uint i = 0,num = 0;i<index_range.size();i++)
    {
        if(i == 2)
            continue;
        Mat img(binary_plate,Rect(index_range[i].x,0,index_range[i].y-index_range[i].x,binary_plate.rows));
        Mat temp_img = Mat::zeros(img.size(),CV_8UC1);
//        cout<<index_range[i].y<<","<<index_range[i].x<<endl;
        int pad = (binary_plate.rows -(index_range[i].y - index_range[i].x))/2;
        img.copyTo(temp_img);
        copyMakeBorder(temp_img,temp_img,0,0,pad,pad, cv::BORDER_CONSTANT,Scalar(0,0,0));

        imwrite(to_string(num++)+".png",temp_img);
        imshow(to_string(i+10),temp_img);
    }
    cout <<"³µÅÆÊ¶±ð½á¹û:";
    for(int i = 0;i<7;i++)
    {
        Mat frame = imread(to_string(i)+".png",1);
        Mat frame_32F;
        frame.convertTo(frame_32F,CV_32FC1);

        Mat blob = blobFromImage(frame_32F/255.0,
                                      1.0,
                                      Size(20,20),
                                      Scalar(0,0,0));

//        cout<<(blob.size);
        net.setInput(blob);
        Mat out = net.forward();
        Point maxclass;
        minMaxLoc(out, NULL, NULL, NULL, &maxclass);
        cout <<label_list[maxclass.x];
    }
    cout<<endl;

#ifdef DEBUG
    imshow("1",license_plate);
    imshow("2",gray_plate);
    imshow("3",binary_plate);
    while(1)
        if(waitKey(0) == 'q')
            break;
#endif
    return 0;
}
