#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

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

Mat cleanUpNoise(Mat noisy_img)
{
  Mat img;
  Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //Maybe make this 3, 3
  erode(noisy_img, img, element);
  dilate(img, img, element);

  return img;
}

std::vector<Point2f> findCentroids(Mat diff)
{
  Mat canny_out;
  Canny(diff, canny_out, 100, 200, 3); //May need to change the middle two values
  std::vector<std::vector<Point>> contours;
  findContours(canny_out, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  std::vector<Moments> mu(contours.size());
  std::vector<Point2f> mc(contours.size());
  for(int i(0); i< contours.size(); i++)
  {
    mu[i] = moments(contours[i]);
    mc[i] = Point2f(static_cast<float>(mu[i].m10/(mu[i].m00+1e-5)),
                    static_cast<float>(mu[i].m01/(mu[i].m00+1e-5)));
  }

  return mc;
}

Mat addTextToImage(Mat gray_img, int num_loops)
{
  std::string label;
  if(num_loops == 1)
    label = "1 Loop";
  else if(num_loops == 2)
    label = "2 Loops";
  else if(num_loops == 3)
    label = "3 Loops";
  else
    label = "";

  putText(gray_img, label.c_str(), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
  //parameters: output, label, origin, font type, font scale, color

  return gray_img;
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
  int key, num_loops(1);

  while(true)
  {
    video >> frame;
    cvtColor(frame, g_frame, COLOR_BGR2GRAY);

    img = binarizeImg(g_frame);
    img = cleanUpNoise(img);
    std::vector<Point2f> centroids = findCentroids(img); //Note: There should only be 1 (for the pretzel)
    //Locate where centroid of pretzel is >100 & < 300
    if(centroids[0].x > 100 && centroids[0].x < 300)
    {
      //Pass image into DNN. Return is an int showing the number of loops
    }
    else
    {
      num_loops = 0;
    }
    g_frame = addTextToImage(g_frame, num_loops);

    imshow("Pretzels", g_frame);

    key = waitKey(10);
    if(key == (int)('q'))
      break;
  }

  return 0;
}
