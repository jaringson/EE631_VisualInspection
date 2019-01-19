#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <vector>

using namespace cv;

int main()
{
  int counter = 0;
  VideoCapture video(1);

  std::string filename = "loop3/3loop";
  std::string filetype = ".png";

  if(!video.isOpened())
      std::cout << "Cannot open camera\n";

  Mat prev_frame;
  while(true)
  {
    //1280x720 resolution
      Mat frame, gray_frame, diff;
      video >> frame;
      cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

     threshold(gray_frame, gray_frame, 180, 255, 0); //For the thresholding part
     Rect roi;
     roi.x = 80;
     roi.width = 420;
     roi.y = 40;
     roi.height = 360;

     gray_frame = gray_frame(roi);
     imshow("Forsgren", gray_frame);
      // counter++;

      if(waitKey(30) >= 0)
      {
        imwrite(filename + std::to_string(counter) + filetype, gray_frame);
        counter++;
      }
  }
  video.release();

  return 0;
}
