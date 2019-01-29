#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

Mat binarizeImg(Mat g_frame)
{
  threshold(g_frame, g_frame, 180, 255, 0); //For the thresholding part
  Rect roi;
  roi.x = 80;
  roi.width = 420;
  roi.y = 40;
  roi.height = 360;

  g_frame = g_frame(roi);

  return g_frame;
}

int main()
{
  VideoCapture video(1);
  if(!video.isOpened())
  {
      std::cout << "Failed to open camera\n";
      return -1;
  }

  Mat frame, g_frame, img;
  int key;

  while(true)
  {
    video >> frame;
    cvtColor(frame, g_frame, COLOR_BGR2GRAY);

    img = binarizeImg(g_frame); //Perform thresholding
    //Locate where centroid of pretzel is
    //if centroid is between low and high pixel location: pass into DNN

    imshow("Pretzels", img);

    key = waitKey(10);
    if(key == (int)('q'))
      break;
  }

  return 0;
}
